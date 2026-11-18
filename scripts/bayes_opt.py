"""Bayesian hyperparameter optimisation (Optuna TPE) for the histogram gradient boosting model.

    python scripts/bayes_opt.py DATA_PROCESSED SUMMARY_CSV OUT_DIR [N_TRIALS]

Objective: mean observation-level Top-1 over 5 trip-grouped CV folds on the
non-quarantined train trips, using the same folds and per-trip point cap as the
untuned run. Student-test is never touched here. Writes bayes_trials_hgb.csv
(every trial) and bayes_best_hgb.json (best configuration and search space).
"""
import json
import os
import sys
import time

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from pathsnap import metrics  # noqa: E402
from pathsnap.features import FEATURE_COLUMNS  # noqa: E402
from pathsnap.modeling import GROUP_OVERRIDES, N_FOLDS, cap_points, load_part  # noqa: E402
from pathsnap.models import SEED  # noqa: E402

SPACE = {
    "learning_rate": "loguniform 0.02 - 0.3",
    "max_leaf_nodes": "int 8 - 128 (log)",
    "min_samples_leaf": "int 10 - 400 (log)",
    "l2_regularization": "loguniform 1e-4 - 10",
    "max_bins": "categorical 64, 128, 255",
    "max_iter": "int 150 - 500",
}


def main():
    proc, summary, out = sys.argv[1:4]
    n_trials = int(sys.argv[4]) if len(sys.argv) > 4 else 40
    s = pd.read_csv(summary, dtype={"trip": str})
    trips = s[(s.partition == "train") & ~s.quarantine].trip.tolist()
    full = load_part(proc, "train", trips)
    groups = pd.factorize(full.trip_id.map(lambda t: GROUP_OVERRIDES.get(t, t)))[0]

    folds = []
    for tr_idx, va_idx in GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(full, groups=groups):
        tr = cap_points(full.iloc[tr_idx]); tr = tr[tr.label >= 0]
        va = full.iloc[va_idx]
        folds.append((tr[FEATURE_COLUMNS].to_numpy(np.float32), tr.label.to_numpy(),
                      va[FEATURE_COLUMNS].to_numpy(np.float32), va))

    def objective(trial):
        cfg = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 8, 128, log=True),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 400, log=True),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
            max_bins=trial.suggest_categorical("max_bins", [64, 128, 255]),
            max_iter=trial.suggest_int("max_iter", 150, 500),
        )
        accs, lls, t0 = [], [], time.time()
        for X, y, Xv, va in folds:
            est = HistGradientBoostingClassifier(**cfg, early_stopping=True, validation_fraction=0.1,
                                                 n_iter_no_change=15, scoring="loss", random_state=SEED)
            est.fit(X, y)
            p = est.predict_proba(Xv)[:, 1]
            accs.append(metrics.top1(va, p)[0])
            lls.append(metrics.candidate_scores(va, p)["log_loss"])
        trial.set_user_attr("cv_log_loss", float(np.mean(lls)))
        trial.set_user_attr("cv_top1_std", float(np.std(accs, ddof=1)))
        trial.set_user_attr("seconds", round(time.time() - t0, 1))
        return float(np.mean(accs))

    # trial 0 is the untuned configuration, so the study contains its own reference point
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.enqueue_trial(dict(learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1e-4,
                             max_bins=255, max_iter=300))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials)
    total = time.time() - t0

    os.makedirs(out, exist_ok=True)
    df = study.trials_dataframe()
    df.round(5).to_csv(os.path.join(out, "bayes_trials_hgb.csv"), index=False)
    best = study.best_trial
    cfg = dict(best.params)
    with open(os.path.join(out, "bayes_best_hgb.json"), "w") as fh:
        json.dump(dict(config=cfg, cv_top1=best.value, trial=best.number, n_trials=n_trials,
                       objective="mean observation-level Top-1 over 5 trip-grouped CV folds",
                       space=SPACE, total_seconds=round(total, 1), seed=SEED), fh, indent=2)
    print("best trial", best.number, "cv top1", round(best.value, 4), cfg)
    print("trial 0 (untuned-like) cv top1", round(study.trials[0].value, 4), "| total seconds", round(total))


if __name__ == "__main__":
    main()
