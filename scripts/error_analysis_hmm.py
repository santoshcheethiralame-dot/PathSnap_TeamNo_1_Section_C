"""What is still wrong after the HMM? Same categories as error_analysis.py, classifier vs classifier + HMM.

    python scripts/error_analysis_hmm.py DATA_RAW PROCESSED PROBS_DIR CACHE_DIR SUMMARY_CSV HMM_SUMMARY_CSV MODEL OUT_CSV

Train trips use out-of-fold probabilities, student-test the final fit. Decoder
settings come from the HMM summary file.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from error_analysis import CATS, classify  # noqa: E402
from run_hmm import load_trip_data  # noqa: E402
from pathsnap.hmm import decode  # noqa: E402
from pathsnap.trips import load_trip  # noqa: E402


def main():
    raw, proc, probs, cache, summary, hsum, model, out = sys.argv[1:9]
    s = pd.read_csv(summary, dtype={"trip": str})
    h = pd.read_csv(hsum).iloc[0]
    params = {k: float(h[k]) for k in ("lam", "beta", "gamma", "penalty")}
    rows = []
    for _, r in s[~s.quarantine].iterrows():
        d = load_trip_data(cache, probs, model, r.partition, r.trip)
        trip = load_trip(os.path.join(raw, r.partition, r.trip), with_route=False)
        f = pd.read_parquet(os.path.join(proc, r.partition, r.trip + ".parquet"), columns=["point_idx", "speed_mps"])
        speed = f.groupby("point_idx").speed_mps.first().reindex(d["pts"]).to_numpy()
        _, idx = decode(d, d["steps"], **params)
        keep = [i for i, l in enumerate(d["label"]) if (l >= 0).any()]
        has_ref = np.array([(d["label"][i] == 1).any() for i in keep])
        ref = np.array([d["arcs"][i][int(np.argmax(d["label"][i]))] if (d["label"][i] == 1).any() else -1 for i in keep])
        for name, pick in (("classifier", [int(np.argmax(d["prob"][i])) for i in keep]), ("hmm", [int(idx[i]) for i in keep])):
            pred = np.array([d["arcs"][i][j] for i, j in zip(keep, pick)])
            wrong = has_ref & (pred != ref)
            cat = np.full(len(keep), "ok", dtype=object)
            cat[~has_ref] = "missing"
            if wrong.any():
                cat[wrong] = classify(trip, ref[wrong], pred[wrong])
            stopped = speed[keep] < 1.0
            rec = dict(method=name, partition=r.partition, trip=r.trip, labelled_points=len(keep), correct=int((cat == "ok").sum()),
                       stopped_points=int(stopped.sum()), stopped_wrong=int((stopped & (cat != "ok")).sum()))
            for c in CATS:
                rec[c] = int((cat == c).sum())
            rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    for part in ("student_test", "train"):
        for name in ("classifier", "hmm"):
            d = df[(df.partition == part) & (df.method == name)]
            n, wrong = int(d.labelled_points.sum()), int(d.labelled_points.sum() - d.correct.sum())
            sp, sw = int(d.stopped_points.sum()), int(d.stopped_wrong.sum())
            print(f"\n{part} / {name}: Top-1 {d.correct.sum() / n * 100:.2f}%  wrong {wrong:,} of {n:,}  "
                  f"stopped error rate {sw / sp * 100:.1f}% (moving {(wrong - sw) / (n - sp) * 100:.1f}%)")
            print("  " + "  ".join(f"{c} {int(d[c].sum()):,} ({d[c].sum() / wrong * 100:.0f}%)" for c in CATS))


if __name__ == "__main__":
    main()
