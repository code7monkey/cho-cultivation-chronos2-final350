"""CHO state forecast utilities. No Google Drive dependency in inference."""
import json, re, zipfile, io
from pathlib import Path
import numpy as np
import pandas as pd

TARGET_MAP = {
    "VCD Mean (cells/mL)": "vcd_cells_ml",
    "Viability Mean (%)": "viability_pct",
    "Glucose (g/L)": "glucose_g_l",
    "Lactate (g/L)": "lactate_g_l",
    "Ammonium (mM)": "ammonium_mm",
}
COV_MAP = {
    "pH": "ph", "Dissolved O2 (% air sat)": "do_pct",
    "Temperature (°C)": "temperature_c", "Liquid volume (L)": "volume_l",
    "Permittivity (pF/cm)": "permittivity_pf_cm",
    "Cumulative feed weight (g)": "feed_cumulative_g",
}
TARGETS = list(TARGET_MAP.values())
COVARIATES = list(COV_MAP.values())
ORIGIN = pd.Timestamp("2000-01-01")

def bin_table(raw, mapping, step_hours):
    missing = set(mapping) | {"Time (h)"}
    missing -= set(raw)
    if missing: raise ValueError(f"Missing source columns: {sorted(missing)}")
    t = pd.to_numeric(raw["Time (h)"], errors="coerce")
    if t.isna().any() or (t < 0).any(): raise ValueError("Invalid elapsed time")
    out = raw[list(mapping)].apply(pd.to_numeric, errors="coerce").rename(columns=mapping)
    # ceil postpones availability to the bin end; never moves observations into the past.
    out["time_h"] = np.ceil(t / step_hours).astype(int) * step_hours
    out = out.sort_values("time_h", kind="stable").groupby("time_h").last()
    return out

def prepare_zip(path, step_hours=6):
    records, inventory = [], []
    with zipfile.ZipFile(path) as z:
        pat = re.compile(r"(EXP\d+)_(fed-batch|batch)_(R\d+)_offline\.csv$")
        for name in sorted(z.namelist()):
            m = pat.search(name)
            if not m: continue
            exp, mode, reactor = m.groups()
            read = lambda n: pd.read_csv(io.BytesIO(z.read(n)), encoding="utf-8-sig")
            offline = bin_table(read(name), TARGET_MAP, step_hours)
            inline_name = name.replace("_offline.csv", "_inline.csv")
            inline = bin_table(read(inline_name), COV_MAP, step_hours)
            # Use only the period for which offline follow-up exists.
            end = int(offline.index.max())
            grid = pd.Index(np.arange(0, end + 1, step_hours), name="time_h")
            frame = offline.reindex(grid).join(inline.reindex(grid))
            # Carry inline values for at most 12h. Targets are never filled here.
            frame[COVARIATES] = frame[COVARIATES].ffill(limit=2)
            frame = frame.reset_index()
            frame["timestamp"] = ORIGIN + pd.to_timedelta(frame["time_h"], unit="h")
            frame["item_id"] = exp + "_" + reactor
            frame["experiment"] = exp
            frame["mode"] = mode
            records.append(frame)
            inventory.append({"item_id": exp+"_"+reactor, "experiment": exp,
                              "mode": mode, "offline_file": name, "inline_file": inline_name})
    if len(records) != 24: raise ValueError(f"Expected 24 cultures, got {len(records)}")
    return pd.concat(records, ignore_index=True), pd.DataFrame(inventory)

def model_frame(df):
    return df[["item_id", "timestamp"] + TARGETS + COVARIATES].copy()

def forecast(pipeline, history, horizon=4, context=48):
    # history must contain only observations available at the forecast cutoff.
    frame = model_frame(history).sort_values(["item_id", "timestamp"])
    result = pipeline.predict_df(frame, target=TARGETS, prediction_length=horizon,
        quantile_levels=[0.1, 0.5, 0.9], context_length=context, batch_size=16,
        cross_learning=False)
    result.columns = [str(c) for c in result.columns]
    return result

def backtest(pipeline, data, horizon=4, context=48, min_past=8):
    rows = []
    for item, group in data.groupby("item_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        for cut in range(min_past, len(group)-horizon+1, horizon):
            history, truth = group.iloc[:cut], group.iloc[cut:cut+horizon]
            preds = forecast(pipeline, history, horizon, context)
            last = history[TARGETS].ffill().iloc[-1]
            for target in TARGETS:
                p = preds[preds.target_name == target].set_index("timestamp")
                for row in truth.itertuples(index=False):
                    y = getattr(row, target)
                    if not np.isfinite(y) or not np.isfinite(last[target]): continue
                    q = p.loc[row.timestamp]
                    rows.append(dict(item_id=item, experiment=row.experiment, mode=row.mode,
                        cutoff=str(history.timestamp.iloc[-1]), timestamp=str(row.timestamp),
                        target=target, y=float(y), persistence=float(last[target]),
                        median=float(q["0.5"]), lower=float(q["0.1"]), upper=float(q["0.9"])))
    if not rows: raise ValueError("No observed evaluation targets")
    return pd.DataFrame(rows)

def metrics(predictions):
    rows=[]
    for (mode, target), g in predictions.groupby(["mode", "target"]):
        for method in ["median", "persistence"]:
            e = g[method]-g.y
            rows.append(dict(mode=mode, target=target, method=method, n=len(g),
                mae=float(e.abs().mean()), rmse=float(np.sqrt((e**2).mean())),
                coverage_80=float(((g.y>=g.lower)&(g.y<=g.upper)).mean()) if method=="median" else None))
    return pd.DataFrame(rows)

def residual_score(p):
    # Heuristic prediction-deviation score, not a contamination probability.
    return (p.y-p["median"]).abs() / np.maximum((p.upper-p.lower)/2, 1e-6)

def harvest_review(predictions, threshold_pct=80.0):
    # Illustrative development trigger, not a validated harvest criterion.
    v = predictions[predictions.target_name == "viability_pct"]
    return {"status": "review" if (v["0.1"] < threshold_pct).any() else "no_trigger",
            "illustrative_viability_threshold_pct": threshold_pct,
            "optimal_harvest_day": None,
            "reason": "Viability forecast only; titer and product-quality labels are unavailable."}

def load_predict(model_dir, history, horizon=None):
    from chronos import Chronos2Pipeline
    import torch
    config = json.loads((Path(model_dir)/"preprocessing.json").read_text())
    pipeline = Chronos2Pipeline.from_pretrained(str(model_dir),
        device_map="cuda" if torch.cuda.is_available() else "cpu")
    return forecast(pipeline, history, horizon or config["horizon"], config["context"])
