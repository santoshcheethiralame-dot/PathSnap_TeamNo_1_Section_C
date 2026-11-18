"""Nearest-arc-with-heading baseline.

    python scripts/run_baseline.py data/processed results/dataset_build_summary.csv results

Picks the heading weight on non-quarantined train trips only, then reports
Top-1 per student_test trip. Writes baseline_weight_grid.csv and
baseline_student_test.csv into the results folder.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from pathsnap.baseline import top1_accuracy  # noqa: E402

WEIGHTS = [0, 5, 10, 20, 40, 80, 160]


def load(proc, part, trips):
    return {t: pd.read_parquet(os.path.join(proc, part, f"{t}.parquet"),
                               columns=["point_idx", "arc_id", "dist_m", "cos_head", "label"]) for t in trips}


def main():
    proc, summary_csv, out = sys.argv[1:4]
    s = pd.read_csv(summary_csv, dtype={"trip": str})
    train = s[(s.partition == "train") & ~s.quarantine].trip.tolist()
    test = s[s.partition == "student_test"].trip.tolist()
    tr, te = load(proc, "train", train), load(proc, "student_test", test)

    grid = []
    for w in WEIGHTS:
        accs = [top1_accuracy(f, w)[0] for f in tr.values()]
        grid.append(dict(heading_weight=w, mean_trip_top1=float(np.nanmean(accs)), n_trips=len(accs)))
    grid = pd.DataFrame(grid)
    best = float(grid.loc[grid.mean_trip_top1.idxmax(), "heading_weight"])
    grid.round(4).to_csv(os.path.join(out, "baseline_weight_grid.csv"), index=False)

    rows = []
    for t, f in te.items():
        acc, n = top1_accuracy(f, best)
        acc0, _ = top1_accuracy(f, 0)
        rows.append(dict(trip=t, labelled_points=n, top1_nearest_only=acc0, top1_with_heading=acc, heading_weight=best))
    res = pd.DataFrame(rows)
    res.round(4).to_csv(os.path.join(out, "baseline_student_test.csv"), index=False)
    print(grid.round(4).to_string(index=False))
    print(res.round(4).to_string(index=False))
    w = res.labelled_points
    print("student_test point-weighted top1: nearest-only", round(float((res.top1_nearest_only * w).sum() / w.sum()), 4),
          "| with heading", round(float((res.top1_with_heading * w).sum() / w.sum()), 4),
          "| per-trip mean", round(float(res.top1_with_heading.mean()), 4))


if __name__ == "__main__":
    main()
