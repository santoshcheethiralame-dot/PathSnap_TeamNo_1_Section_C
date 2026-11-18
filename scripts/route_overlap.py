"""Route overlap on the student-test trips: classifier only vs classifier + HMM.

    python scripts/route_overlap.py DATA_RAW PROCESSED PROBS_DIR OUT_CSV

Overlap is length-weighted against the reference route (arc ids from .route).
The last two columns score our own aligned labels, which is the most any
predictor that picks one arc per fix could reach.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from pathsnap import metrics  # noqa: E402
from pathsnap.labels import align_labels  # noqa: E402
from pathsnap.predict import predict_trip  # noqa: E402
from pathsnap.segments import segment_ids  # noqa: E402
from pathsnap.trips import list_trips, load_trip  # noqa: E402


def main():
    raw, proc, probs, out = sys.argv[1:5]
    rows = []
    for d in list_trips(raw, "student_test"):
        tid = os.path.basename(d)
        trip = load_trip(d)
        seg = segment_ids(trip.t, trip.xy)
        lab, _ = align_labels(trip, seg)
        f = pd.read_parquet(os.path.join(proc, "student_test", tid + ".parquet"))
        f["prob"] = pd.read_parquet(os.path.join(probs, "hgb_tuned", "student_test", tid + ".parquet")).prob.to_numpy()
        clf = metrics.predicted_arcs(f, f.prob.to_numpy(), trip.n_points)
        hmm = np.array(predict_trip(d)["point_arc"])
        row = dict(trip=tid)
        for name, arcs in (("classifier", clf), ("hmm", hmm), ("labels", lab.label_arc.to_numpy())):
            for tag, ign in (("", False), ("_any_dir", True)):
                m = metrics.route_overlap(trip, arcs, trip.route, ignore_direction=ign)
                row[f"{name}{tag}_precision"], row[f"{name}{tag}_recall"], row[f"{name}{tag}_iou"] = m["precision"], m["recall"], m["iou"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.round(4).to_csv(out, index=False)
    print(df[["trip"] + [c for c in df.columns if c.endswith("_iou")]].round(3).to_string(index=False))
    print("mean:", df.drop(columns="trip").mean().round(3).to_dict())


if __name__ == "__main__":
    main()
