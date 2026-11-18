"""Feature table: one row per (point, candidate arc)."""
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "dist_m", "signed_m", "abs_signed_m", "frac", "arc_len_m", "offset_m",
    "cos_head", "abs_head_deg", "head_valid", "speed_mps", "dt_prev_s",
    "rank", "n_cand", "dist_gap_best", "deg_u", "deg_v", "twin_added",
]


def motion(trip, seg_id, window_s=5.0, min_disp_m=5.0):
    """Heading and speed at each fix, from a window of about +/- window_s seconds
    inside the same segment. Falls back to the immediate neighbours when
    sampling is coarse. Heading is only valid if the car moved min_disp_m."""
    n, t, xy = trip.n_points, trip.t, trip.xy
    h = np.zeros((n, 2))
    valid = np.zeros(n, dtype=bool)
    speed = np.zeros(n)
    dt_prev = np.zeros(n)
    dt_prev[1:] = np.diff(t)
    cuts = np.flatnonzero(np.diff(seg_id)) + 1
    for s, e in zip(np.r_[0, cuts], np.r_[cuts, n]):
        m = e - s
        dt_prev[s] = 0.0
        if m < 2:
            continue
        ts, pxy = t[s:e], xy[s:e]
        i = np.arange(m)
        lo = np.minimum(np.searchsorted(ts, ts - window_s, "left"), np.maximum(i - 1, 0))
        hi = np.maximum(np.searchsorted(ts, ts + window_s, "right") - 1, np.minimum(i + 1, m - 1))
        vec = pxy[hi] - pxy[lo]
        disp = np.hypot(vec[:, 0], vec[:, 1])
        speed[s:e] = disp / np.maximum(ts[hi] - ts[lo], 1e-6)
        valid[s:e] = disp >= min_disp_m
        h[s:e] = vec / np.maximum(disp, 1e-9)[:, None]
    return h, valid, speed, dt_prev


def build_features(trip, cands, seg_id, label_arc=None):
    h, valid, speed, dt_prev = motion(trip, seg_id)
    p, a = cands.point_idx.to_numpy(), cands.arc_id.to_numpy()
    cos = (h[p] * trip.arc_dir[a]).sum(1)
    ok = valid[p]
    cos = np.where(ok, np.clip(cos, -1, 1), 0.0)
    best = cands.groupby("point_idx").dist_m.transform("min").to_numpy()
    n_cand = cands.groupby("point_idx").arc_id.transform("size").to_numpy()
    out = pd.DataFrame(
        dict(
            trip_id=trip.trip_id,
            point_idx=cands.point_idx.to_numpy(),
            arc_id=a,
            seg_id=seg_id[p],
            rank=cands["rank"].to_numpy(),
            dist_m=cands.dist_m.to_numpy(),
            signed_m=cands.signed_m.to_numpy(),
            abs_signed_m=np.abs(cands.signed_m.to_numpy()),
            frac=cands.frac.to_numpy(),
            arc_len_m=trip.arc_len[a].astype(np.float32),
            offset_m=(cands.frac.to_numpy() * trip.arc_len[a]).astype(np.float32),
            cos_head=cos.astype(np.float32),
            abs_head_deg=np.where(ok, np.degrees(np.arccos(np.clip(cos, -1, 1))), 90.0).astype(np.float32),
            head_valid=ok.astype(np.int8),
            speed_mps=speed[p].astype(np.float32),
            dt_prev_s=np.clip(dt_prev[p], 0, 60).astype(np.float32),
            n_cand=n_cand.astype(np.int16),
            dist_gap_best=(cands.dist_m.to_numpy() - best).astype(np.float32),
            deg_u=trip.node_degree[trip.u[a]].astype(np.int16),
            deg_v=trip.node_degree[trip.v[a]].astype(np.int16),
            twin_added=cands.twin_added.to_numpy().astype(np.int8),
        )
    )
    if label_arc is not None:
        la = label_arc[p]
        out["label"] = np.where(la < 0, -1, (la == a).astype(np.int8)).astype(np.int8)
    return out
