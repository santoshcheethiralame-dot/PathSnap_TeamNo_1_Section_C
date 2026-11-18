"""Inference entry point used by the API.

Pipeline: load trip (never the reference route) -> segments -> candidates ->
features -> classifier probabilities -> HMM/Viterbi. If models/model.joblib is
missing it falls back to the nearest-arc-with-heading rule so the API still
runs. The returned structure is the same either way.
"""
import json
import os

import joblib
import numpy as np

from .baseline import predict_top1
from .candidates import build_candidates
from .features import build_features
from .hmm import build_graph, build_steps, decode, group_candidates, route_nodes
from .segments import segment_ids
from .trips import load_trip

BASELINE_HEADING_WEIGHT = 10.0   # chosen on non-quarantined train trips, see results/baseline_weight_grid.csv
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models")
_CACHE = {}


def load_artifacts(model_dir=MODEL_DIR):
    key = os.path.abspath(model_dir)
    if key not in _CACHE:
        path = os.path.join(model_dir, "model.joblib")
        if os.path.exists(path):
            with open(os.path.join(model_dir, "config.json")) as fh:
                cfg = json.load(fh)
            _CACHE[key] = (joblib.load(path), cfg)
        else:
            _CACHE[key] = (None, None)
    return _CACHE[key]


def predict_trip(trip_dir, model_dir=MODEL_DIR):
    trip = load_trip(trip_dir, with_route=False)
    seg = segment_ids(trip.t, trip.xy)
    feats = build_features(trip, build_candidates(trip), seg)
    est, cfg = load_artifacts(model_dir)

    arc = np.full(trip.n_points, -1, dtype=np.int64)
    if len(feats):
        if est is None:                                   # fallback: rule baseline
            chosen = predict_top1(feats, BASELINE_HEADING_WEIGHT)
            arc[chosen.point_idx.to_numpy()] = chosen.arc_id.to_numpy()
            method = "nearest-arc + heading rule"
        else:                                             # classifier + HMM
            feats = feats.assign(prob=est.predict_proba(feats[cfg["features"]].to_numpy(np.float32))[:, 1])
            cands = group_candidates(feats)
            graph = build_graph(trip)
            arcs, _ = decode(cands, build_steps(trip, graph, cands, seg), **cfg["hmm"])
            arc[cands["pts"]] = arcs
            method = f"{cfg['model']} classifier + HMM/Viterbi"
    else:
        method = "no candidates"

    used = arc[arc >= 0]
    route = used[np.r_[True, used[1:] != used[:-1]]] if len(used) else used
    nodes = route_nodes(trip, build_graph(trip), arc) if len(used) else np.zeros(0, dtype=np.int64)
    return dict(
        trip_id=trip.trip_id,
        method=method,
        n_points=trip.n_points,
        point_arc=arc.tolist(),                                  # -1 where no candidate was found
        route_arcs=route.tolist(),                               # consecutive duplicates removed
        route_lonlat=trip.to_lonlat(trip.node_xy[nodes]).tolist() if len(nodes) else [],   # joined polyline
        fix_lonlat=trip.to_lonlat(trip.xy).tolist(),
    )
