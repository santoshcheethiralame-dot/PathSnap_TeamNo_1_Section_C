import numpy as np
import pytest

from pathsnap.baseline import top1_accuracy
from pathsnap.candidates import build_candidates
from pathsnap.features import build_features
from pathsnap.labels import align_labels, label_quality
from pathsnap.segments import segment_ids
from pathsnap.trips import PARTITIONS, Trip, list_trips


def straight_road(xy, t, route=None):
    """One straight road along x, nodes every 25 m, an arc in each direction.
    Forward arcs are ids 0-3, reverse arcs are ids 4-7."""
    node_xy = np.array([[25.0 * i, 0.0] for i in range(5)])
    u = np.array([0, 1, 2, 3, 1, 2, 3, 4])
    v = np.array([1, 2, 3, 4, 0, 1, 2, 3])
    return Trip("syn", np.asarray(t, float), np.asarray(xy, float), node_xy, u, v, route)


def test_evaluator_partition_is_unreachable():
    assert "evaluator" not in " ".join(PARTITIONS)
    with pytest.raises(ValueError):
        list_trips(".", "evaluator_test")


def test_lonlat_roundtrip_origin():
    trip = straight_road([[0.0, 0.0], [1.0, 0.0]], [0, 1])
    trip.origin = (77.5, 12.9)
    lonlat = trip.to_lonlat([[0.0, 0.0]])
    assert lonlat[0] == pytest.approx([77.5, 12.9], abs=1e-9)


def test_reverse_twins():
    trip = straight_road([[0, 0], [1, 0]], [0, 1])
    assert list(trip.reverse_arc) == [4, 5, 6, 7, 0, 1, 2, 3]


def test_segments_break_on_gap():
    t = np.array([0, 1, 2, 500, 501])
    xy = np.c_[np.arange(5) * 10.0, np.zeros(5)]
    assert list(segment_ids(t, xy)) == [0, 0, 0, 1, 1]


def test_candidates_include_both_directions():
    trip = straight_road([[30.0, 3.0]], [0.0])
    c = build_candidates(trip, radius=40, k=1)
    assert {1, 5}.issubset(set(c.arc_id))          # arc 1 (+x) and its twin
    near = c[c.arc_id == 1].iloc[0]
    assert near.dist_m == pytest.approx(3.0, abs=1e-4)
    assert near.signed_m == pytest.approx(3.0, abs=1e-4)    # +y is left of +x travel


def test_heading_separates_the_two_directions():
    xy = [[10.0 + 2 * i, 2.0] for i in range(30)]
    trip = straight_road(xy, np.arange(30.0))
    seg = segment_ids(trip.t, trip.xy)
    f = build_features(trip, build_candidates(trip), seg)
    mid = f[(f.point_idx == 15)]
    fwd = mid[mid.arc_id.isin([0, 1, 2, 3])].cos_head
    rev = mid[mid.arc_id.isin([4, 5, 6, 7])].cos_head
    assert (fwd > 0.99).all() and (rev < -0.99).all()


def test_labels_follow_route_through_a_uturn():
    # drive out along arcs 0..3 then back along the twins 7..4
    route = np.array([0, 1, 2, 3, 7, 6, 5, 4])
    out = [[5.0 + 5 * i, 2.0] for i in range(19)]
    back = [[95.0 - 5 * i, -2.0] for i in range(19)]
    xy = np.array(out + back)
    trip = straight_road(xy, np.arange(len(xy), dtype=float), route)
    seg = segment_ids(trip.t, trip.xy)
    labels, stats = align_labels(trip, seg)
    lab = labels.label_arc.to_numpy()
    assert set(lab[:15]) <= {0, 1, 2, 3}
    assert set(lab[-15:]) <= {4, 5, 6, 7}
    assert (np.diff(labels.route_pos.to_numpy()) >= 0).all()
    assert not label_quality(labels, stats)["quarantine"]


def test_far_points_stay_unlabelled_and_quarantine_triggers():
    route = np.array([0, 1, 2, 3])
    xy = np.array([[10.0, 1.0], [30.0, 1.0]] + [[50.0, 500.0]] * 8)
    trip = straight_road(xy, np.arange(10.0), route)
    labels, stats = align_labels(trip, segment_ids(trip.t, trip.xy))
    assert (labels.label_arc.to_numpy()[2:] == -1).all()
    assert label_quality(labels, stats)["quarantine"]


def test_baseline_beats_coin_flip_with_heading():
    # driving in -x, so the reference arcs are the "twin" ids. Ties at equal
    # distance go to the lower arc id, which is the wrong direction here.
    route = np.array([7, 6, 5, 4])
    xy = [[95.0 - 4 * i, 2.0] for i in range(24)]
    trip = straight_road(xy, np.arange(24.0), route)
    seg = segment_ids(trip.t, trip.xy)
    labels, _ = align_labels(trip, seg)
    f = build_features(trip, build_candidates(trip), seg, labels.label_arc.to_numpy())
    acc_no_heading, _ = top1_accuracy(f, 0.0)
    acc_heading, _ = top1_accuracy(f, 50.0)
    assert acc_no_heading < 0.2 and acc_heading > 0.9
