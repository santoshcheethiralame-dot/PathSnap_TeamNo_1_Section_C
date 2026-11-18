"""Where does the HMM help, and where does it not? Per-trip classifier-only vs HMM.

    python scripts/hmm_failure_analysis.py SUMMARY_CSV PROBS_DIR CACHE_DIR OUT_DIR MODEL

Uses the decoder settings saved by run_hmm.py (OUT_DIR/hmm_summary_<MODEL>.csv)
and the cached route distances. Train trips use out-of-fold probabilities.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from run_hmm import load_trip_data, score  # noqa: E402


def main():
    summary, probs, cache, out, model = sys.argv[1:6]
    s = pd.read_csv(summary, dtype={"trip": str})
    h = pd.read_csv(os.path.join(out, f"hmm_summary_{model}.csv")).iloc[0]
    params = {k: float(h[k]) for k in ("lam", "beta", "gamma", "penalty")}
    rows = []
    for _, r in s[~s.quarantine].iterrows():
        d = load_trip_data(cache, probs, model, r.partition, r.trip)
        n, c, e, _ = score(d, params)
        steps = d["steps"]
        rows.append(dict(
            partition=r.partition, trip=r.trip, labelled_points=n, top1_classifier=c / n, top1_hmm=e / n,
            change=(e - c) / n, segments=int(d["seg"].max()) + 1,
            mean_gps_step_m=float(np.mean([s_[1] for s_ in steps if s_ is not None])) if any(s_ is not None for s_ in steps) else np.nan,
            unreachable_share=float(np.mean([np.isinf(s_[0]).mean() for s_ in steps if s_ is not None])),
        ))
    df = pd.DataFrame(rows)
    df.round(5).to_csv(os.path.join(out, f"hmm_per_trip_{model}.csv"), index=False)
    tr = df[df.partition == "train"]
    print("train trips:", len(tr), "| HMM better:", int((tr.change > 0.001).sum()), "| worse:", int((tr.change < -0.001).sum()),
          "| about equal:", int((tr.change.abs() <= 0.001).sum()))
    print("worst 6 train trips for HMM:")
    print(tr.sort_values("change").head(6)[["trip", "labelled_points", "top1_classifier", "top1_hmm", "change", "mean_gps_step_m", "unreachable_share"]].round(4).to_string(index=False))
    print("best 3:")
    print(tr.sort_values("change").tail(3)[["trip", "labelled_points", "top1_classifier", "top1_hmm", "change", "mean_gps_step_m"]].round(4).to_string(index=False))
    print("corr(change, mean_gps_step_m) on train:", round(float(tr[["change", "mean_gps_step_m"]].corr().iloc[0, 1]), 3))


if __name__ == "__main__":
    main()
