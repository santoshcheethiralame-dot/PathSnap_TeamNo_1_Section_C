import numpy as np

from pathsnap.metrics import route_overlap
from pathsnap.trips import Trip


def square_trip():
    nodes = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=float)
    u = np.array([0, 1, 1, 2])
    v = np.array([1, 0, 2, 1])      # arc 1 is the twin of arc 0, arc 3 the twin of arc 2
    return Trip("t", np.arange(3.0), np.zeros((3, 2)), nodes, u, v)


def test_identical_routes_overlap_fully():
    t = square_trip()
    m = route_overlap(t, [0, 2], [0, 2])
    assert m["precision"] == m["recall"] == m["iou"] == 1.0


def test_length_weighting():
    t = square_trip()
    m = route_overlap(t, [0, 2], [0])           # arc 0 is 100 m, arc 2 is 50 m
    assert abs(m["precision"] - 100 / 150) < 1e-9 and m["recall"] == 1.0


def test_direction_matters_unless_ignored():
    t = square_trip()
    assert route_overlap(t, [1], [0])["iou"] == 0.0
    assert route_overlap(t, [1], [0], ignore_direction=True)["iou"] == 1.0
