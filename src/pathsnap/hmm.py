"""HMM map matching with Viterbi decoding.

Hidden state at a GPS fix = one of its candidate arcs.
Emission   = classifier probability of the candidate, normalised over the
             fix's candidates, as lam * log(p).
Transition = -|d_route - d_gps| / (beta + gamma * d_gps), where d_route is the
             driven distance from the previous candidate to this one along the
             directed road graph and d_gps the straight-line distance between
             the two fixes. Unreachable or implausible moves get a flat penalty
             instead of probability zero.
Gaps       = the chain restarts wherever the segment id changes.

Route distances depend only on the road graph and the candidate set, not on the
tuned parameters, so they are computed once per trip (`build_steps`) and reused
for every parameter setting (`decode`).
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

BACKWARD_JITTER_M = 10.0   # moving back this little along one arc is GPS noise, not a loop


def build_graph(trip):
    ok = trip.u != trip.v
    df = pd.DataFrame(dict(u=trip.u[ok], v=trip.v[ok], w=trip.arc_len[ok]))
    df = df.groupby(["u", "v"], as_index=False).w.min()   # parallel arcs: keep the shortest
    n = len(trip.node_xy)
    return csr_matrix((df.w.to_numpy(), (df.u.to_numpy(), df.v.to_numpy())), shape=(n, n))


def group_candidates(df):
    """df needs point_idx, arc_id, frac, prob. Returns per-point arrays."""
    df = df.sort_values("point_idx", kind="stable")
    p = df.point_idx.to_numpy()
    pts, start = np.unique(p, return_index=True)
    end = np.append(start[1:], len(p))
    arcs, frac, prob = df.arc_id.to_numpy(), df.frac.to_numpy(), df.prob.to_numpy()
    return dict(
        pts=pts,
        arcs=[arcs[s:e] for s, e in zip(start, end)],
        frac=[frac[s:e] for s, e in zip(start, end)],
        prob=[prob[s:e] for s, e in zip(start, end)],
    )


def build_steps(trip, graph, cands, seg_id):
    """For each consecutive pair of decoded fixes: None at a segment break, else
    (route-distance matrix [prev candidates x candidates], gps distance)."""
    pts = cands["pts"]
    steps = []
    for k in range(1, len(pts)):
        i0, i1 = pts[k - 1], pts[k]
        if seg_id[i0] != seg_id[i1]:
            steps.append(None)
            continue
        a, b = cands["arcs"][k - 1], cands["arcs"][k]
        fa, fb = cands["frac"][k - 1], cands["frac"][k]
        gps = float(np.hypot(*(trip.xy[i1] - trip.xy[i0])))
        src, dst = np.unique(trip.v[a]), np.unique(trip.u[b])
        sp = dijkstra(graph, directed=True, indices=src, limit=3.0 * gps + 100.0)[:, dst]
        sp = sp[np.searchsorted(src, trip.v[a])][:, np.searchsorted(dst, trip.u[b])]
        la, lb = trip.arc_len[a][:, None], trip.arc_len[b][None, :]
        d = (1 - fa)[:, None] * la + sp + (fb[None, :] * lb)
        fwd = fb[None, :] * lb - fa[:, None] * la
        same = a[:, None] == b[None, :]
        d = np.where(same & (fwd >= -BACKWARD_JITTER_M), np.maximum(fwd, 0.0), d)
        steps.append((d.astype(np.float32), gps))
    return steps


def decode(cands, steps, lam=1.0, beta=10.0, gamma=0.1, penalty=20.0):
    """Viterbi. Returns (arc id, candidate index) for each decoded fix, aligned with cands['pts']."""
    logp = [np.log(np.maximum(p / max(p.sum(), 1e-12), 1e-6)) for p in cands["prob"]]
    n = len(logp)
    dp = lam * logp[0]
    back = [None] * n
    end_dp = {}
    for k in range(1, n):
        s = steps[k - 1]
        if s is None:
            end_dp[k - 1] = dp
            dp = lam * logp[k]
            continue
        d, gps = s
        cost = np.where(np.isfinite(d), np.minimum(np.abs(d - gps) / (beta + gamma * gps), penalty), penalty)
        tot = dp[:, None] - cost
        bi = tot.argmax(0)
        back[k] = bi
        dp = tot[bi, np.arange(tot.shape[1])] + lam * logp[k]
    end_dp[n - 1] = dp

    chosen = np.zeros(n, dtype=np.int64)
    k = n - 1
    while k >= 0:
        j = int(np.argmax(end_dp[k]))
        while True:
            chosen[k] = j
            if k == 0 or back[k] is None:
                break
            j = int(back[k][j])
            k -= 1
        k -= 1
    return np.array([cands["arcs"][i][c] for i, c in enumerate(chosen)], dtype=np.int64), chosen
