"""Overfitting evidence and confusion matrices for the registry models.

    python scripts/model_diagnostics.py DATA_PROCESSED SUMMARY_CSV OUT_DIR model [model ...]

For each model: fit on the capped non-quarantined train trips exactly as in
train_models.py, then score (a) the training rows themselves and (b) student-test.
Writes model_diagnostics.csv (train vs student-test metrics, precision/recall)
and confusion_matrices.csv (student-test, candidate level, threshold 0.5).
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from pathsnap import metrics  # noqa: E402
from pathsnap.modeling import cap_points, fit, load_part, proba  # noqa: E402


def score(df, p):
    m = df.label.to_numpy() >= 0
    y, q = df.label.to_numpy()[m], np.asarray(p)[m]
    tn, fp, fn, tp = confusion_matrix(y, q >= 0.5, labels=[0, 1]).ravel()
    prec, rec, f1, _ = precision_recall_fscore_support(y, q >= 0.5, labels=[1], zero_division=0)
    c = metrics.candidate_scores(df, p)
    return dict(top1=metrics.top1(df, p)[0], macro_f1=c["macro_f1"], log_loss=c["log_loss"],
                precision=float(prec[0]), recall=float(rec[0]), f1_positive=float(f1[0])), (tn, fp, fn, tp)


def main():
    proc, summary, out = sys.argv[1:4]
    names = sys.argv[4:]
    s = pd.read_csv(summary, dtype={"trip": str})
    trips = s[(s.partition == "train") & ~s.quarantine].trip.tolist()
    tests = s[s.partition == "student_test"].trip.tolist()
    tr = cap_points(load_part(proc, "train", trips))
    tr = tr[tr.label >= 0].reset_index(drop=True)
    te = load_part(proc, "student_test", tests)
    rows, cms = [], []
    for name in names:
        est, _ = fit(name, tr)
        a, _ = score(tr, proba(est, tr))
        b, (tn, fp, fn, tp) = score(te, proba(est, te))
        rows.append(dict(model=name, **{f"train_{k}": v for k, v in a.items()}, **{f"student_{k}": v for k, v in b.items()}))
        cms.append(dict(model=name, tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)))
        print(name, {k: round(v, 4) for k, v in rows[-1].items() if k != "model"}, flush=True)
    os.makedirs(out, exist_ok=True)
    for fname, new in (("model_diagnostics.csv", rows), ("confusion_matrices.csv", cms)):
        path = os.path.join(out, fname)
        df = pd.DataFrame(new)
        if os.path.exists(path):
            old = pd.read_csv(path)
            df = pd.concat([old[~old.model.isin(df.model)], df], ignore_index=True)
        df.round(5).to_csv(path, index=False)


if __name__ == "__main__":
    main()
