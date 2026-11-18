"""Per-point training labels from the reference route.

The .route file lists arcs in driving order but says nothing about which arc
each GPS fix belongs to. We recover that with a Viterbi pass over the route
positions: a fix may sit on any route arc within `radius` metres, the position
along the route can only move forward, and the distance travelled along the
route between two fixes should match the straight-line distance between them.
Positions are indices into the route, so an arc that is driven twice (loops,
U-turns) is two different states.
"""
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree


def _point_to_segments(p, a, b):
    ab = b - a
    l2 = np.maximum((ab ** 2).sum(1), 1e-12)
    f = np.clip(((p - a) * ab).sum(1) / l2, 0.0, 1.0)
    c = a + f[:, None] * ab
    return np.hypot(*(p - c).T), f


def align_labels(trip, seg_id, radius=100.0, sigma=10.0, beta=15.0):
    """Returns (labels DataFrame, stats dict). label_arc is -1 where no route
    arc lies within `radius` of the fix."""
    if trip.route is None:
        raise ValueError("trip has no reference route")
    n = trip.n_points
    route = trip.route
    ra = trip.arc_a[route]
    rb = ra + trip.arc_vec[route]
    rl = trip.arc_len[route]
    cum = np.concatenate([[0.0], np.cumsum(rl)[:-1]])

    tree = STRtree(shapely.linestrings(np.stack([ra, rb], axis=1)))
    pi, rj = tree.query(shapely.points(trip.xy), predicate="dwithin", distance=radius)
    order = np.argsort(pi, kind="stable")
    pi, rj = pi[order], rj[order]
    d, f = _point_to_segments(trip.xy[pi], ra[rj], rb[rj])
    s = cum[rj] + f * rl[rj]

    label_arc = np.full(n, -1, dtype=np.int64)
    route_pos = np.full(n, -1, dtype=np.int64)
    label_dist = np.full(n, np.nan)
    stats = dict(n_points=n, n_labelled=0, n_resets=0, disagree_with_nearest=0)
    if len(pi) == 0:
        return pd.DataFrame(dict(label_arc=label_arc, route_pos=route_pos, label_dist=label_dist)), stats

    pts = np.unique(pi)
    starts = np.searchsorted(pi, pts)
    ends = np.append(starts[1:], len(pi))
    J = [rj[a:b] for a, b in zip(starts, ends)]
    D = [d[a:b] for a, b in zip(starts, ends)]
    S = [s[a:b] for a, b in zip(starts, ends)]

    dp = D[0] ** 2 / (2 * sigma ** 2)
    back = []
    for k in range(1, len(pts)):
        gps = float(np.hypot(*(trip.xy[pts[k]] - trip.xy[pts[k - 1]])))
        ds = S[k][None, :] - S[k - 1][:, None]
        tcost = np.abs(np.maximum(ds, 0.0) - gps) / (beta + 0.1 * gps)
        if seg_id[pts[k]] != seg_id[pts[k - 1]]:
            tcost = np.zeros_like(tcost)           # time gap: no distance evidence
        forward = J[k][None, :] >= J[k - 1][:, None]
        total = dp[:, None] + np.where(forward, tcost, np.inf)
        best = total.argmin(0)
        val = total[best, np.arange(len(best))]
        if not np.isfinite(val).any():             # nothing reachable going forward
            stats["n_resets"] += 1
            total = dp[:, None] + tcost
            best = total.argmin(0)
            val = total[best, np.arange(len(best))]
        back.append(best)
        dp = D[k] ** 2 / (2 * sigma ** 2) + val

    pick = int(np.argmin(dp))
    chosen = [pick]
    for bp in reversed(back):
        pick = int(bp[pick])
        chosen.append(pick)
    chosen = chosen[::-1]

    for k, c in enumerate(chosen):
        i = pts[k]
        label_arc[i] = route[J[k][c]]
        route_pos[i] = J[k][c]
        label_dist[i] = D[k][c]
        if c != int(np.argmin(D[k])):
            stats["disagree_with_nearest"] += 1
    stats["n_labelled"] = len(pts)
    return pd.DataFrame(dict(label_arc=label_arc, route_pos=route_pos, label_dist=label_dist)), stats


def label_quality(labels, stats, min_labelled_frac=0.85, max_median_dist=25.0):
    """Summary plus the quarantine decision for a trip."""
    lab = labels[labels.label_arc >= 0]
    n = stats["n_points"]
    out = dict(
        n_points=n,
        labelled_frac=stats["n_labelled"] / n,
        label_dist_median=float(lab.label_dist.median()) if len(lab) else np.nan,
        label_dist_p95=float(lab.label_dist.quantile(0.95)) if len(lab) else np.nan,
        frac_label_dist_gt50=float((lab.label_dist > 50).mean()) if len(lab) else np.nan,
        n_resets=stats["n_resets"],
        disagree_with_nearest_frac=stats["disagree_with_nearest"] / max(stats["n_labelled"], 1),
    )
    out["quarantine"] = bool(
        out["labelled_frac"] < min_labelled_frac
        or not (out["label_dist_median"] <= max_median_dist)
    )
    return out
