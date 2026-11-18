"""Load a PathSnap trip and put everything in a local metric frame."""
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from pyproj import Transformer

# The evaluator partition is deliberately absent: development code cannot
# list or open it by name.
PARTITIONS = ("train", "student_test")


def list_trips(root, partition):
    if partition not in PARTITIONS:
        raise ValueError(f"partition must be one of {PARTITIONS}, got {partition!r}")
    base = os.path.join(root, partition)
    return sorted(os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))


def _reverse_arcs(u, v, n_nodes):
    key = u.astype(np.int64) * n_nodes + v
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    want = v.astype(np.int64) * n_nodes + u
    pos = np.clip(np.searchsorted(sorted_key, want), 0, len(sorted_key) - 1)
    hit = (sorted_key[pos] == want) & (u != v)
    return np.where(hit, order[pos], -1)


@dataclass
class Trip:
    trip_id: str
    t: np.ndarray              # (n,) seconds
    xy: np.ndarray             # (n, 2) metres in the trip's local frame
    node_xy: np.ndarray        # (m, 2) metres
    u: np.ndarray              # (a,) from-node of each arc
    v: np.ndarray              # (a,) to-node of each arc
    route: Optional[np.ndarray] = None   # reference arc ids; None at inference
    origin: Optional[tuple] = None       # (lon0, lat0) of the local frame

    def __post_init__(self):
        self.arc_a = self.node_xy[self.u]
        self.arc_vec = self.node_xy[self.v] - self.arc_a
        self.arc_len = np.maximum(np.hypot(self.arc_vec[:, 0], self.arc_vec[:, 1]), 1e-6)
        self.arc_dir = self.arc_vec / self.arc_len[:, None]
        self.reverse_arc = _reverse_arcs(self.u, self.v, len(self.node_xy))
        self.node_degree = np.bincount(np.concatenate([self.u, self.v]), minlength=len(self.node_xy))

    def to_lonlat(self, xy):
        """Local metres back to (lon, lat) degrees, for maps and the UI."""
        lon0, lat0 = self.origin
        inv = Transformer.from_crs(
            f"+proj=tmerc +lat_0={lat0} +lon_0={lon0} +k=1 +units=m +datum=WGS84", 4326, always_xy=True
        )
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        lon, lat = inv.transform(xy[:, 0], xy[:, 1])
        return np.c_[lon, lat]

    @property
    def n_points(self):
        return len(self.t)

    @property
    def n_arcs(self):
        return len(self.u)


def load_trip(trip_dir, with_route=True):
    """Read the four files of one trip. Pass with_route=False for inference,
    so the reference route is never opened."""
    tid = os.path.basename(os.path.normpath(trip_dir))
    p = os.path.join(trip_dir, tid)
    trk = pd.read_csv(p + ".track", sep=r"\s+", header=None, names=["lon", "lat", "t"])
    nod = pd.read_csv(p + ".nodes", sep=r"\s+", header=None, names=["lon", "lat"])
    arc = pd.read_csv(p + ".arcs", sep=r"\s+", header=None, names=["u", "v"])
    route = None
    if with_route and os.path.exists(p + ".route"):
        route = pd.read_csv(p + ".route", header=None)[0].to_numpy(dtype=np.int64)

    lon0, lat0 = float(trk.lon.mean()), float(trk.lat.mean())
    tf = Transformer.from_crs(
        4326, f"+proj=tmerc +lat_0={lat0} +lon_0={lon0} +k=1 +units=m +datum=WGS84", always_xy=True
    )
    px, py = tf.transform(trk.lon.to_numpy(), trk.lat.to_numpy())
    nx, ny = tf.transform(nod.lon.to_numpy(), nod.lat.to_numpy())
    return Trip(
        trip_id=tid,
        t=trk.t.to_numpy(dtype=float),
        xy=np.c_[px, py],
        node_xy=np.c_[nx, ny],
        u=arc.u.to_numpy(dtype=np.int64),
        v=arc.v.to_numpy(dtype=np.int64),
        route=route,
        origin=(lon0, lat0),
    )
