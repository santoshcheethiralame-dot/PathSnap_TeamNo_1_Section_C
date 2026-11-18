"""Per-trip data audit for the PathSnap dataset.

Usage:
    python scripts/audit_data.py DATA_ROOT results/data_audit_per_trip.csv

DATA_ROOT holds train/ and student_test/, each with one folder per trip
containing <id>.track, <id>.nodes, <id>.arcs and <id>.route.

Never point this at the evaluator partition. Everything here uses a
proxy label (nearest route arc), not the final label alignment.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from shapely import STRtree

RADII = [15, 25, 40, 60, 100]
KS = [1, 3, 5, 10, 20]
PARTS = ("train", "student_test")


def load_trip(d):
    tid = os.path.basename(d)
    p = os.path.join(d, tid)
    trk = pd.read_csv(p + ".track", sep=r"\s+", header=None, names=["lon", "lat", "t"])
    nod = pd.read_csv(p + ".nodes", sep=r"\s+", header=None, names=["lon", "lat"])
    arc = pd.read_csv(p + ".arcs", sep=r"\s+", header=None, names=["u", "v"])
    rte = pd.read_csv(p + ".route", header=None, names=["a"])["a"].to_numpy()
    return tid, trk, nod, arc, rte


def local_projection(trk):
    # trips are worldwide, so project each one around its own centroid
    crs = f"+proj=tmerc +lat_0={trk.lat.mean()} +lon_0={trk.lon.mean()} +k=1 +units=m +datum=WGS84"
    return Transformer.from_crs(4326, crs, always_xy=True)


def nearest_dist(tree, pts):
    idx, dist = tree.query_nearest(pts, return_distance=True, all_matches=False)
    pos = np.empty(len(pts), dtype=int)
    d = np.empty(len(pts))
    pos[idx[0]] = idx[1]
    d[idx[0]] = dist
    return pos, d


def audit_trip(part, d):
    tid, trk, nod, arc, rte = load_trip(d)
    r = dict(part=part, trip=tid, n_pts=len(trk), n_nodes=len(nod), n_arcs=len(arc), n_route=len(rte))
    r["nan_values"] = int(trk.isna().sum().sum() + nod.isna().sum().sum())
    r["lon_mean"], r["lat_mean"] = float(trk.lon.mean()), float(trk.lat.mean())

    dt = np.diff(trk.t.to_numpy())
    r.update(t_start=float(trk.t.iloc[0]), dt_eq1=float((dt == 1).mean()), dt_max=float(dt.max()),
             dt_gt1=int((dt > 1).sum()), dt_le0=int((dt <= 0).sum()))

    n = len(nod)
    r["route_ids_valid"] = bool(rte.min() >= 0 and rte.max() < len(arc))
    r["arc_node_ids_valid"] = bool(arc.u.max() < n and arc.v.max() < n)
    deg = np.bincount(np.concatenate([arc.u, arc.v]), minlength=n)
    r["isolated_nodes"] = int((deg == 0).sum())
    r["self_loops"] = int((arc.u == arc.v).sum())
    u, v = arc.u.to_numpy().astype(np.int64), arc.v.to_numpy().astype(np.int64)
    r["arcs_with_reverse_twin"] = float(np.isin(v * n + u, u * n + v).mean())

    ru, rv = u[rte], v[rte]
    r["route_chain_breaks"] = int((rv[:-1] != ru[1:]).sum())
    r["route_repeated_arcs"] = int(len(rte) - len(set(rte)))

    tf = local_projection(trk)
    nx, ny = tf.transform(nod.lon.to_numpy(), nod.lat.to_numpy())
    px, py = tf.transform(trk.lon.to_numpy(), trk.lat.to_numpy())
    N = np.c_[nx, ny]
    seg = shapely.linestrings(np.stack([N[u], N[v]], axis=1))
    alen = shapely.length(seg)
    r["arc_len_median_m"] = float(np.median(alen))
    r["route_len_m"] = float(alen[rte].sum())
    step = np.hypot(np.diff(px), np.diff(py))
    r["track_len_m"] = float(step.sum())
    r["step_max_m"] = float(step.max())
    r["steps_gt50m"] = int((step > 50).sum())
    r["track_in_network_bbox"] = float(((px >= nx.min()) & (px <= nx.max()) & (py >= ny.min()) & (py <= ny.max())).mean())

    pts = shapely.points(px, py)
    tree_all = STRtree(seg)
    tree_route = STRtree(seg[rte])

    _, d_any = nearest_dist(tree_all, pts)
    lab_pos, d_route = nearest_dist(tree_route, pts)
    r["frac_within15m_any_arc"] = float((d_any < 15).mean())
    r["frac_within15m_route"] = float((d_route < 15).mean())
    r["route_dist_median_m"] = float(np.median(d_route))
    r["route_dist_p95_m"] = float(np.percentile(d_route, 95))
    r["route_dist_max_m"] = float(d_route.max())
    r["proxy_label_backsteps"] = int((np.diff(lab_pos) < 0).sum())

    lab = rte[lab_pos]
    mv = np.c_[np.diff(px), np.diff(py)]
    mv = np.vstack([mv, mv[-1]])
    ad = N[v[lab]] - N[u[lab]]
    cos = (mv * ad).sum(1) / (np.linalg.norm(mv, axis=1) * np.linalg.norm(ad, axis=1) + 1e-9)
    moving = np.linalg.norm(mv, axis=1) > 3
    r["stationary_frac"] = float(1 - moving.mean())
    r["label_heading_agree"] = float((cos[moving] > 0).mean()) if moving.any() else np.nan

    pi, ai = tree_all.query(pts, predicate="dwithin", distance=max(RADII))
    df = pd.DataFrame(dict(p=pi, a=ai, d=shapely.distance(pts[pi], seg[ai])))
    df["is_label"] = df.a.to_numpy() == lab[df.p.to_numpy()]
    df["rank"] = df.groupby("p").d.rank(method="first")
    for R in RADII:
        sub = df[df.d <= R]
        cnt = sub.groupby("p").size().reindex(range(len(pts)), fill_value=0)
        hit = sub.groupby("p").is_label.any().reindex(range(len(pts)), fill_value=False)
        r[f"recall_R{R}"] = float(hit.mean())
        r[f"ncand_R{R}"] = float(cnt.mean())
        r[f"empty_R{R}"] = float((cnt == 0).mean())
    lr = df[df.is_label]
    for K in KS:
        r[f"recall_top{K}_nearest"] = float(lr[lr["rank"] <= K].p.nunique() / len(pts))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    a = ap.parse_args()
    rows = []
    for part in PARTS:
        for d in sorted(glob.glob(os.path.join(a.root, part, "*"))):
            rows.append(audit_trip(part, d))
            print(part, rows[-1]["trip"], rows[-1]["n_pts"], flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    pd.DataFrame(rows).round(4).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
