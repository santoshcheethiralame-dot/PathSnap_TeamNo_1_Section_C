"""Training harness: trip-grouped cross-validation, final fit, student-test evaluation.

Rules baked in: only non-quarantined train trips are used for fitting and CV,
student-test is only ever predicted (never fitted on), the number of points per
trip is capped so long trips cannot dominate, everything is seeded.
"""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from . import metrics
from .features import FEATURE_COLUMNS
from .models import REGISTRY, SEED
from .segments import segment_ids
from .trips import list_trips, load_trip

MAX_POINTS_PER_TRIP = 1000
N_FOLDS = 5
# trips from the same city share roads, so they must stay in the same fold
GROUP_OVERRIDES = {"00000010": "moscow", "00000016": "moscow", "00000029": "moscow"}


def load_part(proc, part, trips):
    return pd.concat(
        [pd.read_parquet(os.path.join(proc, part, f"{t}.parquet")) for t in trips], ignore_index=True
    )


def cap_points(df, max_points=MAX_POINTS_PER_TRIP, seed=SEED):
    rng = np.random.default_rng(seed)
    parts = []
    for _, g in df.groupby("trip_id", sort=False):
        pts = g.point_idx.unique()
        if len(pts) > max_points:
            g = g[np.isin(g.point_idx.to_numpy(), rng.choice(pts, max_points, replace=False))]
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def xy(df):
    return df[FEATURE_COLUMNS].to_numpy(dtype=np.float32), df.label.to_numpy()


def fit(name, train_df):
    spec = REGISTRY[name]
    est = spec["build"](spec["config"])
    X, y = xy(train_df)
    t0 = time.time()
    est.fit(X, y)
    return est, time.time() - t0


def proba(est, df):
    return est.predict_proba(df[FEATURE_COLUMNS].to_numpy(dtype=np.float32))[:, 1]


def run_model(name, proc, summary_csv, raw_root, out_dir, probs_dir=None, log=print):
    s = pd.read_csv(summary_csv, dtype={"trip": str})
    train_trips = s[(s.partition == "train") & ~s.quarantine].trip.tolist()
    test_trips = s[s.partition == "student_test"].trip.tolist()
    full = load_part(proc, "train", train_trips)          # every point, for held-out evaluation
    train_all = full[full.label >= 0]
    groups = full.trip_id.map(lambda t: GROUP_OVERRIDES.get(t, t))

    # ---- grouped CV ----
    folds, oof = [], np.full(len(full), np.nan)
    gkf = GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    trip_codes = pd.factorize(groups)[0]
    for k, (tr_idx, va_idx) in enumerate(gkf.split(full, groups=trip_codes)):
        tr = cap_points(full.iloc[tr_idx]); tr = tr[tr.label >= 0]
        va = full.iloc[va_idx]
        est, t_fit = fit(name, tr)
        p = proba(est, va)
        oof[va_idx] = p
        sc = metrics.candidate_scores(va, p)
        t1, npts = metrics.top1(va, p)
        folds.append(dict(model=name, fold=k, n_train_rows=len(tr), n_val_trips=va.trip_id.nunique(),
                          n_val_points=npts, top1=t1, fit_seconds=round(t_fit, 1), **sc))
        log(name, "fold", k, {k2: round(v, 4) for k2, v in folds[-1].items() if isinstance(v, float)})

    # ---- final fit on all train, evaluate on student-test ----
    tr = cap_points(full); tr = tr[tr.label >= 0]
    est, t_final = fit(name, tr)
    te = load_part(proc, "student_test", test_trips)
    t0 = time.time()
    p_te = proba(est, te)
    t_inf = time.time() - t0
    sc = metrics.candidate_scores(te, p_te)
    t1, npts = metrics.top1(te, p_te)

    per_trip = []
    for t in test_trips:
        m = (te.trip_id == t).to_numpy()
        d = te[m]
        trip = load_trip(os.path.join(raw_root, "student_test", t))
        seg = segment_ids(trip.t, trip.xy)
        arcs = metrics.predicted_arcs(d, p_te[m], trip.n_points)
        c = metrics.candidate_scores(d, p_te[m])
        a, n = metrics.top1(d, p_te[m])
        per_trip.append(dict(model=name, trip=t, n_points=n, top1=a, macro_f1=c["macro_f1"], log_loss=c["log_loss"],
                             route_continuity=metrics.route_continuity(trip, arcs, seg)))
    pt = pd.DataFrame(per_trip)
    spec = REGISTRY[name]
    f = pd.DataFrame(folds)
    row = dict(
        model=name, family=spec["family"], config=json.dumps(spec["config"], default=str), seed=SEED,
        max_points_per_trip=MAX_POINTS_PER_TRIP, n_train_trips=full.trip_id.nunique(), n_train_rows=len(tr),
        cv_folds=N_FOLDS, cv_macro_f1_mean=f.macro_f1.mean(), cv_macro_f1_std=f.macro_f1.std(),
        cv_top1_mean=f.top1.mean(), cv_log_loss_mean=f.log_loss.mean(),
        cv_fit_seconds_mean=f.fit_seconds.mean(), final_fit_seconds=round(t_final, 1),
        student_top1=t1, student_macro_f1=sc["macro_f1"], student_log_loss=sc["log_loss"],
        student_route_continuity=float(pt.route_continuity.mean()),
        student_top1_per_trip_mean=float(pt.top1.mean()),
        inference_seconds_per_trip=t_inf / len(test_trips), inference_us_per_row=1e6 * t_inf / len(te),
    )

    # ---- write results ----
    os.makedirs(out_dir, exist_ok=True)
    _replace_rows(os.path.join(out_dir, "model_runs.csv"), "model", name, pd.DataFrame([row]))
    _replace_rows(os.path.join(out_dir, "cv_folds.csv"), "model", name, f)
    _replace_rows(os.path.join(out_dir, "student_test_per_trip.csv"), "model", name, pt)
    if spec["history"] is not None:
        with open(os.path.join(out_dir, f"history_{name}.json"), "w") as fh:
            json.dump(spec["history"](est), fh)

    if probs_dir:   # out-of-fold probabilities for train, final-model probabilities for student-test
        for part, df, pr in (("train", full, oof), ("student_test", te, p_te)):
            for t, g in df.assign(prob=pr).groupby("trip_id"):
                os.makedirs(os.path.join(probs_dir, name, part), exist_ok=True)
                g[["point_idx", "arc_id", "prob"]].to_parquet(os.path.join(probs_dir, name, part, f"{t}.parquet"), index=False)
    return row


def _replace_rows(path, key, value, new):
    if os.path.exists(path):
        old = pd.read_csv(path, dtype={"trip": str})
        new = pd.concat([old[old[key] != value], new], ignore_index=True)
    new.round(5).to_csv(path, index=False)
