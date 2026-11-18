"""Fit the chosen classifier on all non-quarantined train trips and save it for the API.

    python scripts/export_model.py DATA_PROCESSED SUMMARY_CSV HMM_SUMMARY_CSV MODEL_NAME

Writes models/model.joblib and models/config.json (feature list, decoder
settings, seed, training set). Student-test and evaluator data are not used.
"""
import json
import os
import sys

import joblib
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from pathsnap.features import FEATURE_COLUMNS  # noqa: E402
from pathsnap.modeling import MAX_POINTS_PER_TRIP, cap_points, fit, load_part  # noqa: E402
from pathsnap.models import REGISTRY, SEED  # noqa: E402


def main():
    proc, summary, hmm_csv, name = sys.argv[1:5]
    s = pd.read_csv(summary, dtype={"trip": str})
    trips = s[(s.partition == "train") & ~s.quarantine].trip.tolist()
    full = load_part(proc, "train", trips)
    tr = cap_points(full)
    tr = tr[tr.label >= 0]
    est, secs = fit(name, tr)
    h = pd.read_csv(hmm_csv).iloc[0]
    out = os.path.join(ROOT, "models")
    os.makedirs(out, exist_ok=True)
    joblib.dump(est, os.path.join(out, "model.joblib"), compress=3)
    cfg = dict(
        model=name, model_config=REGISTRY[name]["config"], features=FEATURE_COLUMNS, seed=SEED,
        hmm=dict(lam=float(h.lam), beta=float(h.beta), gamma=float(h.gamma), penalty=float(h.penalty)),
        candidates=dict(radius_m=40.0, k=10), max_points_per_trip=MAX_POINTS_PER_TRIP,
        trained_on=f"{len(trips)} non-quarantined train trips, {len(tr)} candidate rows", fit_seconds=round(secs, 1),
    )
    with open(os.path.join(out, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    print("saved", round(os.path.getsize(os.path.join(out, "model.joblib")) / 1e6, 2), "MB", cfg["hmm"])


if __name__ == "__main__":
    main()
