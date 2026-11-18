"""Evaluation metrics. Definitions are proposals until the organisers confirm
them (docs/05_open_questions.md #4 and #5)."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss


def point_key(df):
    codes = df.trip_id.astype("category").cat.codes.to_numpy().astype(np.int64)
    return codes * 1_000_000 + df.point_idx.to_numpy().astype(np.int64)


def top1(df, prob):
    """Observation-level Top-1 over labelled points: the highest-scoring
    candidate of each point is compared with the reference arc. A point whose
    reference arc is not among the candidates counts as wrong."""
    d = pd.DataFrame(dict(k=point_key(df), p=prob, y=df.label.to_numpy()))
    d = d[d.y >= 0]
    best = d.groupby("k")["p"].idxmax()
    return float(d.loc[best, "y"].mean()), int(len(best))


def candidate_scores(df, prob, threshold=0.5):
    """Candidate-level binary metrics (is-reference vs not) on labelled points."""
    m = df.label.to_numpy() >= 0
    y, p = df.label.to_numpy()[m], np.asarray(prob)[m]
    return dict(
        macro_f1=float(f1_score(y, p >= threshold, average="macro")),
        log_loss=float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
    )


def route_continuity(trip, point_arc, seg_id):
    """Fraction of consecutive distinct arcs, within one segment, where the end
    node of one is the start node of the next. Points without an arc (-1) are
    skipped. NaN if there are no such pairs."""
    arc = np.asarray(point_arc)
    ok = arc >= 0
    arc, seg = arc[ok], np.asarray(seg_id)[ok]
    if len(arc) < 2:
        return float("nan")
    change = (arc[1:] != arc[:-1]) & (seg[1:] == seg[:-1])
    a, b = arc[:-1][change], arc[1:][change]
    if len(a) == 0:
        return float("nan")
    return float((trip.v[a] == trip.u[b]).mean())


def predicted_arcs(df, prob, n_points):
    """Highest-scoring arc per point as an array of length n_points (-1 = none)."""
    d = pd.DataFrame(dict(p=df.point_idx.to_numpy(), s=prob, a=df.arc_id.to_numpy()))
    best = d.groupby("p")["s"].idxmax()
    out = np.full(n_points, -1, dtype=np.int64)
    out[d.loc[best, "p"].to_numpy()] = d.loc[best, "a"].to_numpy()
    return out


def route_overlap(trip, pred_arcs, ref_arcs, ignore_direction=False):
    """Length-weighted overlap between a predicted route and the reference
    route, both given as arc ids. Precision: share of the predicted length that
    lies on the reference. Recall: share of the reference length that was
    predicted. IoU: shared length over combined length. With ignore_direction
    an arc and its opposite twin count as the same road."""
    def keyset(arcs):
        a = np.unique(np.asarray(arcs, dtype=np.int64))
        a = a[a >= 0]
        if ignore_direction:
            twin = trip.reverse_arc[a]
            a = np.unique(np.where((twin >= 0) & (twin < a), twin, a))
        return a
    p, r = keyset(pred_arcs), keyset(ref_arcs)
    both = np.intersect1d(p, r)
    lp, lr, lb = trip.arc_len[p].sum(), trip.arc_len[r].sum(), trip.arc_len[both].sum()
    union = lp + lr - lb
    return dict(precision=float(lb / lp) if lp else float("nan"), recall=float(lb / lr) if lr else float("nan"),
                iou=float(lb / union) if union else float("nan"))
