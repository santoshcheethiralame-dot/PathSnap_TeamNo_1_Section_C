"""Inference entry point used by the API.

Today this runs the nearest-arc-with-heading baseline. When the trained
classifier and the HMM exist, only the marked block changes; the returned
structure stays the same so the API and UI keep working.

The reference route is never opened here (with_route=False).
"""
import numpy as np

from .baseline import predict_top1
from .candidates import build_candidates
from .features import build_features
from .segments import segment_ids
from .trips import load_trip

BASELINE_HEADING_WEIGHT = 10.0   # chosen on non-quarantined train trips, see results/baseline_weight_grid.csv


def predict_trip(trip_dir, heading_weight=BASELINE_HEADING_WEIGHT):
    trip = load_trip(trip_dir, with_route=False)
    seg = segment_ids(trip.t, trip.xy)
    feats = build_features(trip, build_candidates(trip), seg)

    # --- replace this block with classifier scores + HMM decoding ---
    chosen = predict_top1(feats, heading_weight)
    # -----------------------------------------------------------------

    arc = np.full(trip.n_points, -1, dtype=np.int64)
    arc[chosen.point_idx.to_numpy()] = chosen.arc_id.to_numpy()
    used = arc[arc >= 0]
    route = used[np.r_[True, used[1:] != used[:-1]]] if len(used) else used
    if len(route):
        nodes = np.concatenate([trip.u[route], trip.v[route][-1:]])
        route_lonlat = trip.to_lonlat(trip.node_xy[nodes])
    else:
        route_lonlat = np.zeros((0, 2))
    return dict(
        trip_id=trip.trip_id,
        n_points=trip.n_points,
        point_arc=arc.tolist(),                      # -1 where no candidate was found
        route_arcs=route.tolist(),                   # consecutive duplicates removed
        route_lonlat=route_lonlat.tolist(),          # [[lon, lat], ...] polyline for the map
        fix_lonlat=trip.to_lonlat(trip.xy).tolist(),
    )
