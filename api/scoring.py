"""Reference-based evaluation. This module is used only for scoring."""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pathsnap.trips import load_trip
from pathsnap.predict import predict_trip
from pathsnap.segments import segment_ids
from pathsnap.labels import align_labels
from pathsnap.metrics import route_continuity


def score_trip(trip_dir, prediction=None):
    """
    Score one trip using its reference route.

    Reference labels are loaded only inside this scoring module.
    The API can pass an existing prediction to avoid predicting twice.
    """

    trip_dir = Path(trip_dir)

    # Load reference route for scoring only
    trip = load_trip(trip_dir, with_route=True)

    # Predict only if a prediction was not supplied
    if prediction is None:
        prediction = predict_trip(trip_dir)

    predicted_arc = np.asarray(prediction["point_arc"])

    if len(predicted_arc) != trip.n_points:
        raise ValueError("Prediction and reference trip lengths do not match.")

    # Generate continuous segment IDs
    seg_id = segment_ids(trip.t, trip.xy)

    # Align GPS points with reference route
    labels, stats = align_labels(trip, seg_id)
    reference_arc = labels["label_arc"].to_numpy()

    # Evaluate only points with reference labels
    labelled = reference_arc >= 0
    n_labelled = int(labelled.sum())

    if n_labelled > 0:
        top1_accuracy = float(
            (predicted_arc[labelled] == reference_arc[labelled]).mean()
        )
    else:
        top1_accuracy = None

    # Fraction of all GPS points that received a prediction
    prediction_coverage = float((predicted_arc >= 0).mean())

    # Route continuity
    continuity = route_continuity(trip, predicted_arc, seg_id)

    if not np.isfinite(continuity):
        continuity = None

    return {
        "n_points": int(trip.n_points),
        "n_labelled": n_labelled,
        "top1_accuracy": top1_accuracy,
        "prediction_coverage": prediction_coverage,
        "route_continuity": continuity,
        "predicted_route_arcs": int(len(prediction["route_arcs"]))
    }
