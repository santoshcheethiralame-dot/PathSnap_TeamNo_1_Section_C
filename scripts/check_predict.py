"""End-to-end check of predict_trip against our own label alignment.

    python scripts/check_predict.py DATA_RAW

Runs predict_trip exactly as the API would (reference route never opened by the
predictor), then scores it with labels from align_labels, as the scoring module
will. Student-test trips only.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from pathsnap import metrics  # noqa: E402
from pathsnap.labels import align_labels  # noqa: E402
from pathsnap.predict import load_artifacts, predict_trip  # noqa: E402
from pathsnap.segments import segment_ids  # noqa: E402
from pathsnap.trips import list_trips, load_trip  # noqa: E402


def main():
    raw = sys.argv[1]
    print("method:", "classifier + HMM" if load_artifacts()[0] is not None else "rule baseline fallback")
    rows = []
    for d in list_trips(raw, "student_test"):
        t0 = time.time()
        pred = predict_trip(d)
        secs = time.time() - t0
        trip = load_trip(d)
        seg = segment_ids(trip.t, trip.xy)
        lab, _ = align_labels(trip, seg)
        la, pa = lab.label_arc.to_numpy(), np.array(pred["point_arc"])
        m = la >= 0
        rows.append(dict(trip=os.path.basename(d), labelled=int(m.sum()), top1=float((pa[m] == la[m]).mean()),
                         route_continuity=metrics.route_continuity(trip, pa, seg), seconds=round(secs, 2)))
    df = pd.DataFrame(rows)
    print(df.round(4).to_string(index=False))
    w = df.labelled
    print("point-weighted Top-1", round(float((df.top1 * w).sum() / w.sum()), 4),
          "| mean route continuity", round(float(df.route_continuity.mean()), 4),
          "| max seconds per trip", df.seconds.max())


if __name__ == "__main__":
    main()
