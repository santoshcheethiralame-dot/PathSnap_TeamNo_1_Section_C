"""What kind of mistakes does the tuned boosting classifier make, and why is trip 16 weak?

    python scripts/error_analysis.py DATA_RAW PROCESSED PROBS_DIR SUMMARY_CSV OUT_DIR

Every labelled fix whose top-scored candidate is not the reference arc is put in
one category (first match wins):
  opposite     the reference road driven the other way (the twin)
  next_arc     the arc just before or after the reference arc on the same road
  sibling      another arc leaving or entering the same junction node
  parallel     not connected, but within 20 m of the reference arc
  elsewhere    anything else
  missing      the reference arc is not among the candidates
Train trips use out-of-fold probabilities; student-test trips use the final fit.
"""
import os
import sys

import numpy as np
import pandas as pd
import shapely

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from pathsnap.trips import load_trip  # noqa: E402

CATS = ["opposite", "next_arc", "sibling", "parallel", "elsewhere", "missing"]


def classify(trip, ref, pred):
    cat = np.full(len(ref), "elsewhere", dtype=object)
    done = np.zeros(len(ref), dtype=bool)

    def put(mask, name):
        nonlocal done
        m = mask & ~done
        cat[m] = name
        done |= m
    put(trip.reverse_arc[ref] == pred, "opposite")
    put((trip.v[ref] == trip.u[pred]) | (trip.v[pred] == trip.u[ref]), "next_arc")
    put((trip.u[ref] == trip.u[pred]) | (trip.v[ref] == trip.v[pred]), "sibling")
    rest = ~done
    if rest.any():
        a = shapely.linestrings(np.stack([trip.arc_a[ref[rest]], trip.arc_a[ref[rest]] + trip.arc_vec[ref[rest]]], axis=1))
        b = shapely.linestrings(np.stack([trip.arc_a[pred[rest]], trip.arc_a[pred[rest]] + trip.arc_vec[pred[rest]]], axis=1))
        near = np.zeros(len(ref), dtype=bool)
        near[np.flatnonzero(rest)] = shapely.distance(a, b) < 20.0
        put(near, "parallel")
    return cat


def main():
    raw, proc, probs, summary, out = sys.argv[1:6]
    s = pd.read_csv(summary, dtype={"trip": str})
    rows = []
    for _, r in s[~s.quarantine].iterrows():
        trip = load_trip(os.path.join(raw, r.partition, r.trip), with_route=False)
        f = pd.read_parquet(os.path.join(proc, r.partition, r.trip + ".parquet"))
        f["prob"] = pd.read_parquet(os.path.join(probs, "hgb_tuned", r.partition, r.trip + ".parquet")).prob.to_numpy()
        f = f[f.label >= 0]
        best = f.loc[f.groupby("point_idx").prob.idxmax()].set_index("point_idx")
        has_ref = f[f.label == 1].groupby("point_idx").arc_id.first()
        pts = best.index.to_numpy()
        ref = has_ref.reindex(pts)
        n = len(pts)
        missing = ref.isna().to_numpy()
        ref_a = ref.fillna(-1).to_numpy(dtype=np.int64)
        pred_a = best.arc_id.to_numpy(dtype=np.int64)
        wrong = (ref_a != pred_a)
        cat = np.full(n, "ok", dtype=object)
        cat[missing] = "missing"
        w = wrong & ~missing
        if w.any():
            cat[w] = classify(trip, ref_a[w], pred_a[w])
        speed = best.speed_mps.to_numpy()
        stopped = speed < 1.0
        ref_dist = f[f.label == 1].groupby("point_idx").dist_m.first().reindex(pts).to_numpy()
        rec = dict(partition=r.partition, trip=r.trip, labelled_points=n, top1=float((cat == "ok").mean()),
                   median_dist_to_reference_m=float(np.nanmedian(ref_dist)), share_stopped=float(stopped.mean()),
                   error_share_stopped=float(stopped[wrong].mean()) if wrong.any() else np.nan,
                   median_n_candidates=float(best.n_cand.median()), n_segments=int(best.seg_id.nunique()))
        for c in CATS:
            rec[c] = int((cat == c).sum())
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.round(5).to_csv(os.path.join(out, "error_analysis_per_trip.csv"), index=False)

    def pooled(d, label):
        wrong = int(d[CATS].sum().sum())
        print(f"\n{label}: {len(d)} trips, {int(d.labelled_points.sum()):,} fixes, {wrong:,} wrong "
              f"({wrong / d.labelled_points.sum() * 100:.1f}%), stopped share {np.average(d.share_stopped, weights=d.labelled_points) * 100:.1f}% of fixes, "
              f"{np.nansum(d.error_share_stopped * d[CATS].sum(axis=1)) / max(wrong, 1) * 100:.1f}% of errors")
        print((d[CATS].sum() / wrong * 100).round(1).to_string())
    pooled(df[df.partition == "student_test"], "student-test")
    pooled(df[df.partition == "train"], "train out-of-fold")
    t16 = df[df.trip == "00000016"]
    print("\ntrip 16:\n", t16.T.to_string())
    tr = df[df.partition == "train"]
    print("\ncorrelation with Top-1 over train trips:")
    for c in ("median_dist_to_reference_m", "share_stopped", "median_n_candidates", "n_segments", "labelled_points"):
        print(f"  {c}: {tr[[c, 'top1']].corr(method='spearman').iloc[0, 1]:.2f}")
    print("\nlowest 6 train trips by Top-1:")
    print(tr.sort_values("top1").head(6)[["trip", "labelled_points", "top1", "median_dist_to_reference_m", "share_stopped", "n_segments"] + CATS].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
