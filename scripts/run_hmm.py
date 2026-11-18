"""Tune the HMM decoder on train out-of-fold probabilities, report on student-test.

    python scripts/run_hmm.py DATA_RAW DATA_PROCESSED SUMMARY_CSV PROBS_DIR CACHE_DIR OUT_DIR MODEL

Route distances (the slow part) are cached per trip in CACHE_DIR and do not
depend on the model or on the decoder parameters. Decoder parameters are
chosen on train only (probabilities are out-of-fold, so the classifier never
saw those trips); student-test is only used for reporting.
"""
import itertools
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from pathsnap import metrics  # noqa: E402
from pathsnap.hmm import build_graph, build_steps, decode, group_candidates  # noqa: E402
from pathsnap.segments import segment_ids  # noqa: E402
from pathsnap.trips import load_trip  # noqa: E402

GRID = dict(lam=[0.5, 1.0, 2.0], beta=[2.0, 3.0, 5.0, 7.0, 10.0, 20.0], gamma=[0.0, 0.1, 0.3], penalty=[20.0])
DATA = {}


def build_cache(args):
    raw, proc, cache, part, tid = args
    path = os.path.join(cache, part, tid + ".pkl")
    if os.path.exists(path):
        return tid
    trip = load_trip(os.path.join(raw, part, tid), with_route=False)
    seg = segment_ids(trip.t, trip.xy)
    f = pd.read_parquet(os.path.join(proc, part, tid + ".parquet"), columns=["point_idx", "arc_id", "frac", "label"])
    f["prob"] = 1.0
    f = f.sort_values("point_idx", kind="stable")
    cands = group_candidates(f)
    steps = build_steps(trip, build_graph(trip), cands, seg)
    p = f.point_idx.to_numpy()
    _, start = np.unique(p, return_index=True)
    end = np.append(start[1:], len(p))
    lab = f.label.to_numpy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(dict(pts=cands["pts"], arcs=cands["arcs"], frac=cands["frac"], steps=steps,
                         label=[lab[s:e] for s, e in zip(start, end)], seg=seg, n_points=trip.n_points), fh)
    return tid


def load_trip_data(cache, probs, model, part, tid):
    with open(os.path.join(cache, part, tid + ".pkl"), "rb") as fh:
        d = pickle.load(fh)
    pr = pd.read_parquet(os.path.join(probs, model, part, tid + ".parquet")).sort_values("point_idx", kind="stable")
    prob = pr.prob.to_numpy()
    sizes = [len(a) for a in d["arcs"]]
    cuts = np.cumsum(sizes)[:-1]
    assert np.array_equal(pr.arc_id.to_numpy(), np.concatenate(d["arcs"])), f"{tid}: prob rows do not match candidates"
    d["prob"] = np.split(prob, cuts)
    return d


def score(d, params):
    """Returns (labelled points, correct classifier-only, correct HMM) and the HMM arcs per point."""
    arcs, idx = decode(d, d["steps"], **params)
    cl = np.array([np.argmax(p) for p in d["prob"]])
    lab = d["label"]
    mask = np.array([(l >= 0).any() for l in lab])
    ok_h = np.array([l[i] == 1 for l, i in zip(lab, idx)])
    ok_c = np.array([l[i] == 1 for l, i in zip(lab, cl)])
    return int(mask.sum()), int((ok_c & mask).sum()), int((ok_h & mask).sum()), arcs


def init_worker(cache, probs, model, trips):
    for part, tid in trips:
        DATA[(part, tid)] = load_trip_data(cache, probs, model, part, tid)


def eval_params(params):
    n = c = h = 0
    for d in DATA.values():
        a, b, e, _ = score(d, params)
        n, c, h = n + a, c + b, h + e
    return dict(params, points=n, top1_classifier=c / n, top1_hmm=h / n)


def main():
    raw, proc, summary, probs, cache, out, model = sys.argv[1:8]
    s = pd.read_csv(summary, dtype={"trip": str})
    train = [("train", t) for t in s[(s.partition == "train") & ~s.quarantine].trip]
    test = [("student_test", t) for t in s[s.partition == "student_test"].trip]
    os.makedirs(out, exist_ok=True)

    with Pool(10) as pool:
        for _ in pool.imap_unordered(build_cache, [(raw, proc, cache, p, t) for p, t in train + test]):
            pass
        print("route distances cached", flush=True)
        combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    with Pool(10, initializer=init_worker, initargs=(cache, probs, model, train)) as pool:
        grid = pd.DataFrame(pool.map(eval_params, combos))
    grid.round(5).to_csv(os.path.join(out, f"hmm_grid_{model}.csv"), index=False)
    best = grid.loc[grid.top1_hmm.idxmax()]
    params = {k: float(best[k]) for k in GRID}
    print("best params (train OOF)", params, "classifier-only", round(best.top1_classifier, 4),
          "hmm", round(best.top1_hmm, 4), flush=True)

    rows = []
    for part, tid in test:
        d = load_trip_data(cache, probs, model, part, tid)
        n, c, h, arcs = score(d, params)
        trip = load_trip(os.path.join(raw, part, tid), with_route=False)
        full = np.full(d["n_points"], -1, dtype=np.int64)
        full[d["pts"]] = arcs
        cl_full = np.full(d["n_points"], -1, dtype=np.int64)
        cl_full[d["pts"]] = [a[np.argmax(p)] for a, p in zip(d["arcs"], d["prob"])]
        rows.append(dict(
            model=model, trip=tid, labelled_points=n, top1_classifier=c / n, top1_hmm=h / n,
            route_continuity_classifier=metrics.route_continuity(trip, cl_full, d["seg"]),
            route_continuity_hmm=metrics.route_continuity(trip, full, d["seg"]),
        ))
    res = pd.DataFrame(rows)
    res.round(5).to_csv(os.path.join(out, f"hmm_student_test_{model}.csv"), index=False)
    w = res.labelled_points
    wavg = lambda col: float((res[col] * w).sum() / w.sum())
    summ = dict(model=model, **params,
                train_oof_top1_classifier=float(best.top1_classifier), train_oof_top1_hmm=float(best.top1_hmm),
                student_top1_classifier=wavg("top1_classifier"), student_top1_hmm=wavg("top1_hmm"),
                student_route_continuity_classifier=float(res.route_continuity_classifier.mean()),
                student_route_continuity_hmm=float(res.route_continuity_hmm.mean()),
                student_trips_hmm_better=int((res.top1_hmm > res.top1_classifier).sum()),
                student_trips_hmm_worse=int((res.top1_hmm < res.top1_classifier).sum()))
    pd.DataFrame([summ]).round(5).to_csv(os.path.join(out, f"hmm_summary_{model}.csv"), index=False)
    print(res.round(4).to_string(index=False))
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in summ.items()}, flush=True)


if __name__ == "__main__":
    main()
