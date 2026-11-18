"""Build the per-trip candidate/feature tables (train and student_test only).

    python scripts/build_dataset.py data/raw data/processed results/dataset_build_summary.csv [trip_id ...]

Writes data/processed/<partition>/<trip_id>.parquet and one summary row per trip.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from pathsnap.candidates import build_candidates  # noqa: E402
from pathsnap.features import build_features  # noqa: E402
from pathsnap.labels import align_labels, label_quality  # noqa: E402
from pathsnap.segments import segment_ids  # noqa: E402
from pathsnap.trips import PARTITIONS, list_trips, load_trip  # noqa: E402

RADIUS, K = 40.0, 10


def main():
    root, out_dir, summary_csv = sys.argv[1:4]
    only = set(sys.argv[4:])
    rows = []
    for part in PARTITIONS:
        os.makedirs(os.path.join(out_dir, part), exist_ok=True)
        for d in list_trips(root, part):
            tid = os.path.basename(d)
            if only and tid not in only:
                continue
            t0 = time.time()
            trip = load_trip(d)
            seg = segment_ids(trip.t, trip.xy)
            labels, stats = align_labels(trip, seg)
            q = label_quality(labels, stats)
            cands = build_candidates(trip, radius=RADIUS, k=K)
            feats = build_features(trip, cands, seg, labels.label_arc.to_numpy())
            feats.to_parquet(os.path.join(out_dir, part, tid + ".parquet"), index=False)

            lab_pts = labels.label_arc.to_numpy() >= 0
            hit = feats[feats.label == 1].point_idx.nunique()
            miss_rate = 1 - hit / max(lab_pts.sum(), 1)
            rows.append(dict(
                partition=part, trip=tid, **q,
                n_segments=int(seg.max()) + 1,
                n_cand_rows=len(feats),
                mean_cand_per_point=len(feats) / trip.n_points,
                points_without_candidate=int(trip.n_points - feats.point_idx.nunique()),
                candidate_miss_rate=float(miss_rate),
                seconds=round(time.time() - t0, 1),
            ))
            print(part, tid, trip.n_points, f"lab={q['labelled_frac']:.2f}", f"miss={miss_rate:.3f}",
                  "QUARANTINE" if q["quarantine"] else "", f"{rows[-1]['seconds']}s", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(summary_csv)), exist_ok=True)
    pd.DataFrame(rows).round(4).to_csv(summary_csv, index=False)


if __name__ == "__main__":
    main()
