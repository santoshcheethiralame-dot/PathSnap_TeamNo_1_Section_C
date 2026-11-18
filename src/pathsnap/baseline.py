"""Nearest-arc-with-heading rule: the yardstick every model must beat."""
import numpy as np


def predict_top1(feats, w):
    """Pick, per point, the candidate minimising dist_m + w * (1 - cos_head)."""
    score = feats.dist_m.to_numpy() + w * (1.0 - feats.cos_head.to_numpy())
    best = feats.assign(_s=score).groupby("point_idx")["_s"].idxmin()
    cols = [c for c in ("point_idx", "arc_id", "label") if c in feats.columns]
    return feats.loc[best, cols]


def top1_accuracy(feats, w):
    """Observation-level Top-1 over points that have a label."""
    pred = predict_top1(feats, w)
    pred = pred[pred.label >= 0]
    return float(pred.label.mean()) if len(pred) else np.nan, len(pred)
