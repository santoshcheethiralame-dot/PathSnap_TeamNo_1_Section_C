"""Context features: heading carried over a stop, and movement along each candidate arc."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap.candidates import build_candidates
from pathsnap.features import build_features
from pathsnap.segments import segment_ids
from pathsnap.trips import Trip


def long_road(n_nodes=41):
    """A straight road along x, nodes every 25 m. Forward arcs 0..n-2, twins n-1..2n-3."""
    node_xy = np.array([[25.0 * i, 0.0] for i in range(n_nodes)])
    fwd = [(i, i + 1) for i in range(n_nodes - 1)]
    u = np.array([a for a, _ in fwd] + [b for _, b in fwd])
    v = np.array([b for _, b in fwd] + [a for a, _ in fwd])
    return node_xy, u, v


def drive_then_stop():
    """20 s eastwards at 10 m/s, then 20 s standing still at x = 200, 2 m north of the road."""
    t = np.arange(40, dtype=float)
    x = np.where(t < 20, 10.0 * t, 200.0)
    xy = np.c_[x, np.full(40, 2.0)]
    node_xy, u, v = long_road()
    return Trip("syn", t, xy, node_xy, u, v)


def features_of(trip):
    seg = segment_ids(trip.t, trip.xy)
    return build_features(trip, build_candidates(trip), seg), len(trip.u) // 2


def test_heading_is_carried_over_a_stop():
    trip = drive_then_stop()
    f, n_fwd = features_of(trip)
    stopped = f[(f.point_idx == 35) & (f.arc_id < n_fwd)]      # forward arcs only
    twin = f[(f.point_idx == 35) & (f.arc_id >= n_fwd)]
    assert (stopped.head_valid == 0).all() and (stopped.cos_head == 0).all()      # the plain heading is lost
    assert (stopped.cos_head_ff > 0.99).all() and (stopped.head_filled == 1).all()
    assert (twin.cos_head_ff < -0.99).all()
    assert stopped.head_age_s.min() > 0


def test_movement_along_an_arc_tells_the_direction():
    trip = drive_then_stop()
    f, n_fwd = features_of(trip)
    mid = f[f.point_idx == 10]
    fwd, rev = mid[mid.arc_id < n_fwd], mid[mid.arc_id >= n_fwd]
    assert (fwd.along_b5 > 40).all() and (fwd.along_f5 > 40).all()      # about 50 m in 5 s
    assert (rev.along_b5 < -40).all() and (rev.along_f5 < -40).all()


def test_fixes_near_the_segment_edge_use_the_edge():
    trip = drive_then_stop()
    f, _ = features_of(trip)
    first = f[f.point_idx == 0]
    assert (first.along_b5 == 0).all() and (first.along_b15 == 0).all()      # nothing before the first fix
    assert not f.isna().any().any()


def test_distance_of_neighbouring_fixes_to_the_arc():
    trip = drive_then_stop()
    f, n_fwd = features_of(trip)
    row = f[(f.point_idx == 10) & (f.arc_id == 4)]      # the arc covering x 100 to 125; the fix 5 s ago was at x = 50
    assert len(row) == 1
    assert abs(float(row.d_b5.iloc[0]) - np.hypot(50.0, 2.0)) < 0.5
