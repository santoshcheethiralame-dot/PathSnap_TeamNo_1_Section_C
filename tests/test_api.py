"""API tests on a small synthetic trip, so they run without the organiser data."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
from fastapi.testclient import TestClient  # noqa: E402

import api.main as main  # noqa: E402


def write_trip(base, tid, with_route=True):
    """A straight east-west road of 8 nodes (about 110 m apart), both directions, driven eastwards."""
    d = base / tid
    d.mkdir(parents=True)
    lon = 77.0 + 0.001 * np.arange(8)
    n = 8
    fwd = [(i, i + 1) for i in range(n - 1)]
    arcs = fwd + [(v, u) for u, v in fwd]
    (d / f"{tid}.nodes").write_text("\n".join(f"{x:.6f} 12.000000" for x in lon))
    (d / f"{tid}.arcs").write_text("\n".join(f"{u} {v}" for u, v in arcs))
    rng = np.random.default_rng(0)
    xs = np.linspace(lon[0] + 0.0002, lon[-1] - 0.0002, 60)
    ys = 12.0 + rng.normal(0, 0.00002, len(xs))
    (d / f"{tid}.track").write_text("\n".join(f"{x:.6f} {y:.6f} {i}" for i, (x, y) in enumerate(zip(xs, ys))))
    if with_route:
        (d / f"{tid}.route").write_text("\n".join(str(i) for i in range(n - 1)))
    return d


@pytest.fixture()
def client(tmp_path, monkeypatch):
    write_trip(tmp_path, "00000001")
    write_trip(tmp_path, "00000002", with_route=False)
    (tmp_path / "not_a_trip").mkdir()
    monkeypatch.setattr(main, "TRIPS_DIR", tmp_path)
    return TestClient(main.app)


def test_health_and_trip_list(client):
    assert client.get("/health").json()["status"] == "ok"
    body = client.get("/trips").json()
    assert body == {"count": 2, "trips": ["00000001", "00000002"]}


def test_predict_returns_route_and_scores(client):
    r = client.post("/predict", json={"trip_id": "00000001"})
    assert r.status_code == 200
    j = r.json()
    assert j["scored"] is True and j["n_labelled"] > 0
    assert j["top1_accuracy"] > 0.9
    assert j["prediction_coverage"] == 1.0
    assert len(j["fix_lonlat"]) == j["n_points"] == len(j["point_arc"])
    assert len(j["route_lonlat"]) >= 2
    assert "route" not in j and "reference" not in " ".join(j.keys())   # ground truth never leaves the server


def test_trip_without_reference_is_not_scored_but_still_served(client):
    j = client.post("/predict", json={"trip_id": "00000002"}).json()
    assert j["scored"] is False and j["top1_accuracy"] is None
    assert len(j["route_lonlat"]) >= 2


def test_bad_requests(client):
    assert client.post("/predict", json={"trip_id": "../00000001"}).status_code == 400
    assert client.post("/predict", json={"trip_id": "99999999"}).status_code == 404
    assert client.post("/predict", json={"trip_id": "not_a_trip"}).status_code == 404
    assert client.post("/predict", json={}).status_code == 422


def test_empty_folder_lists_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "TRIPS_DIR", tmp_path / "missing")
    assert TestClient(main.app).get("/trips").json() == {"count": 0, "trips": []}
