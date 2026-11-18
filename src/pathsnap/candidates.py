"""Candidate arcs for every GPS point."""
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree


def arc_geometry(trip, p, a):
    """Distance, signed lateral offset and position along arc for (point, arc) pairs.
    signed > 0 means the point is to the left of the arc's direction of travel."""
    A, V, L = trip.arc_a[a], trip.arc_vec[a], trip.arc_len[a]
    w = trip.xy[p] - A
    f = np.clip((w * V).sum(1) / (L ** 2), 0.0, 1.0)
    c = A + f[:, None] * V
    dist = np.hypot(*(trip.xy[p] - c).T)
    signed = (V[:, 0] * w[:, 1] - V[:, 1] * w[:, 0]) / L
    return dist, signed, f


def build_candidates(trip, radius=40.0, k=10, widen=2.0):
    """For each point: the k nearest arcs within `radius` (retry at radius*widen
    if there are none), plus the reverse twin of each kept arc so both
    directions of a road are always present."""
    n, n_arcs = trip.n_points, trip.n_arcs
    tree = STRtree(shapely.linestrings(np.stack([trip.arc_a, trip.arc_a + trip.arc_vec], axis=1)))
    pts = shapely.points(trip.xy)
    pi, ai = tree.query(pts, predicate="dwithin", distance=radius)
    have = np.zeros(n, dtype=bool)
    have[pi] = True
    miss = np.flatnonzero(~have)
    if len(miss):
        p2, a2 = tree.query(pts[miss], predicate="dwithin", distance=radius * widen)
        pi, ai = np.concatenate([pi, miss[p2]]), np.concatenate([ai, a2])
    if len(pi) == 0:
        return _empty()

    dist, signed, frac = arc_geometry(trip, pi, ai)
    o = np.lexsort((ai, dist, pi))
    pi, ai, dist, signed, frac = pi[o], ai[o], dist[o], signed[o], frac[o]
    rank = np.arange(len(pi)) - np.searchsorted(pi, pi)
    keep = rank < k
    pi, ai, dist, signed, frac = pi[keep], ai[keep], dist[keep], signed[keep], frac[keep]
    twin = np.zeros(len(pi), dtype=bool)

    rev = trip.reverse_arc[ai]
    key = pi.astype(np.int64) * n_arcs + ai
    new_key = pi.astype(np.int64) * n_arcs + rev
    add = (rev >= 0) & ~np.isin(new_key, key)
    if add.any():
        _, first = np.unique(new_key[add], return_index=True)   # drop duplicate twins
        idx = np.flatnonzero(add)[first]
        tp, ta = pi[idx], rev[idx]
        td, ts, tf = arc_geometry(trip, tp, ta)
        pi = np.concatenate([pi, tp])
        ai = np.concatenate([ai, ta])
        dist = np.concatenate([dist, td])
        signed = np.concatenate([signed, ts])
        frac = np.concatenate([frac, tf])
        twin = np.concatenate([twin, np.ones(len(tp), dtype=bool)])

    o = np.lexsort((ai, twin, dist, pi))
    pi, ai, dist, signed, frac, twin = pi[o], ai[o], dist[o], signed[o], frac[o], twin[o]
    rank = np.arange(len(pi)) - np.searchsorted(pi, pi)
    return pd.DataFrame(
        dict(
            point_idx=pi.astype(np.int32),
            arc_id=ai.astype(np.int64),
            rank=rank.astype(np.int16),
            dist_m=dist.astype(np.float32),
            signed_m=signed.astype(np.float32),
            frac=frac.astype(np.float32),
            twin_added=twin,
        )
    )


def _empty():
    return pd.DataFrame(
        dict(
            point_idx=np.array([], dtype=np.int32), arc_id=np.array([], dtype=np.int64),
            rank=np.array([], dtype=np.int16), dist_m=np.array([], dtype=np.float32),
            signed_m=np.array([], dtype=np.float32), frac=np.array([], dtype=np.float32),
            twin_added=np.array([], dtype=bool),
        )
    )
