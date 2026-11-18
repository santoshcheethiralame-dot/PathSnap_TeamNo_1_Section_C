
"""Generate exploratory data analysis figures for PathSnap.

Usage:
    python scripts/make_eda.py data/raw results/data_audit_per_trip.csv report/figures/eda
"""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib.collections import LineCollection
from shapely import STRtree

# Allow importing the project package from src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pathsnap.trips import list_trips, load_trip


# ---------------------------------------------------------
# 1. LOAD AUDIT DATA
# ---------------------------------------------------------

def load_audit(path):
    """Load the existing per-trip data audit."""
    return pd.read_csv(path)


# ---------------------------------------------------------
# 2. MISSING VALUE ANALYSIS
# ---------------------------------------------------------

def plot_missing_values(audit, output_dir):
    """Plot missing-value counts for each partition."""

    summary = audit.groupby("part")["nan_values"].sum()

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        summary.index,
        summary.values,
        width=0.5,
    )

    max_count = max(float(summary.max()), 1.0)
    ax.set_ylim(0, max_count * 1.25)

    for bar, value in zip(bars, summary.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(float(value), 0) + max_count * 0.04,
            f"{int(value)} missing values",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_title("Missing Values by Partition")
    ax.set_xlabel("Dataset Partition")
    ax.set_ylabel("Missing Values")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        output_dir / "missing_values.png",
        dpi=160,
    )
    plt.close(fig)


# ---------------------------------------------------------
# 3. GPS TIME-GAP ANALYSIS
# ---------------------------------------------------------

def plot_time_gaps(root, output_dir):
    """Plot GPS observation time gaps for both partitions.

    Only the central 99% of observations are displayed.
    Observations outside this range are counted and reported.
    Original data is not modified.
    """

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, partition in zip(
        axes,
        ["train", "student_test"],
    ):
        gaps = []

        for trip_dir in list_trips(root, partition):
            trip = load_trip(
                trip_dir,
                with_route=False,
            )

            dt = np.diff(trip.t)
            gaps.extend(dt[dt > 0])

        gaps = np.asarray(gaps, dtype=float)
        gaps = gaps[np.isfinite(gaps)]

        if len(gaps) == 0:
            ax.set_title(
                f"Time Gaps - {partition} (no data)"
            )
            continue

        cutoff = np.percentile(gaps, 99)

        visible = gaps[gaps <= cutoff]
        excluded = len(gaps) - len(visible)

        ax.hist(
            visible,
            bins=50,
        )

        ax.set_title(f"Time Gaps - {partition}")
        ax.set_xlabel("Time gap (seconds)")
        ax.set_ylabel("Frequency")

        ax.text(
            0.97,
            0.95,
            f"99th percentile: {cutoff:.1f} s\n"
            f"Excluded from view: {excluded}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )

        ax.grid(axis="y", alpha=0.2)

    fig.suptitle(
        "GPS Observation Time-Gap Distribution"
    )

    fig.tight_layout()
    fig.savefig(
        output_dir / "time_gaps.png",
        dpi=160,
    )
    plt.close(fig)


# ---------------------------------------------------------
# 4. GPS-TO-REFERENCE-ROUTE DISTANCE ANALYSIS
# ---------------------------------------------------------

def plot_route_distances(root, output_dir):
    """Calculate and plot GPS-to-reference-route distances.

    Only the central 99% of observations are displayed.
    The number of observations outside the displayed range
    is reported on each subplot.
    """

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, partition in zip(
        axes,
        ["train", "student_test"],
    ):
        distances = []

        for trip_dir in list_trips(root, partition):
            trip = load_trip(
                trip_dir,
                with_route=True,
            )

            if (
                trip.route is None
                or len(trip.route) == 0
            ):
                continue

            segments = shapely.linestrings(
                np.stack(
                    [
                        trip.node_xy[trip.u],
                        trip.node_xy[trip.v],
                    ],
                    axis=1,
                )
            )

            route_segments = segments[trip.route]

            tree = STRtree(route_segments)

            points = shapely.points(
                trip.xy[:, 0],
                trip.xy[:, 1],
            )

            _, d = tree.query_nearest(
                points,
                return_distance=True,
                all_matches=False,
            )

            distances.extend(d)

        distances = np.asarray(
            distances,
            dtype=float,
        )

        distances = distances[
            np.isfinite(distances)
        ]

        if len(distances) == 0:
            ax.set_title(
                f"GPS-to-Route Distance - {partition} (no data)"
            )
            continue

        cutoff = np.percentile(
            distances,
            99,
        )

        visible = distances[
            distances <= cutoff
        ]

        excluded = len(distances) - len(visible)

        ax.hist(
            visible,
            bins=60,
        )

        ax.set_title(
            f"GPS-to-Route Distance - {partition}"
        )

        ax.set_xlabel(
            "Distance to reference route (metres)"
        )

        ax.set_ylabel("GPS observations")
        ax.set_xlim(left=0)

        ax.text(
            0.97,
            0.95,
            f"99th percentile: {cutoff:.1f} m\n"
            f"Excluded from view: {excluded}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )

        ax.grid(axis="y", alpha=0.2)

    fig.suptitle(
        "GPS-to-Reference-Route Distance Distribution"
    )

    fig.tight_layout()
    fig.savefig(
        output_dir / "gps_route_distance.png",
        dpi=160,
    )
    plt.close(fig)


# ---------------------------------------------------------
# 5. GPS TRAJECTORY VISUALIZATION
# ---------------------------------------------------------

def plot_trajectory(trip_dir, partition, output_dir):
    """Plot one complete GPS trajectory."""

    trip = load_trip(
        trip_dir,
        with_route=False,
    )

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(
        trip.xy[:, 0],
        trip.xy[:, 1],
        linewidth=0.8,
        marker=".",
        markersize=2,
        label="GPS trajectory",
    )

    ax.set_title(
        f"GPS Trajectory - {partition} - Trip {trip.trip_id}"
    )

    ax.set_xlabel("Local X (metres)")
    ax.set_ylabel("Local Y (metres)")

    ax.set_aspect(
        "equal",
        adjustable="datalim",
    )

    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir / f"{partition}_{trip.trip_id}.png",
        dpi=160,
    )

    plt.close(fig)


# ---------------------------------------------------------
# 6. ROAD NETWORK VISUALIZATION
# ---------------------------------------------------------

def plot_road_network(trip_dir, partition, output_dir):
    """Plot road-network arcs with GPS observations."""

    trip = load_trip(
        trip_dir,
        with_route=False,
    )

    segments = np.stack(
        [
            trip.node_xy[trip.u],
            trip.node_xy[trip.v],
        ],
        axis=1,
    )

    fig, ax = plt.subplots(figsize=(9, 8))

    collection = LineCollection(
        segments,
        linewidths=0.35,
        alpha=0.5,
        label="Road network",
    )

    ax.add_collection(collection)

    ax.scatter(
        trip.xy[:, 0],
        trip.xy[:, 1],
        s=3,
        label="GPS observations",
        zorder=3,
    )

    ax.autoscale()

    ax.set_aspect(
        "equal",
        adjustable="datalim",
    )

    ax.set_title(
        f"Road Network and GPS - {partition} - Trip {trip.trip_id}"
    )

    ax.set_xlabel("Local X (metres)")
    ax.set_ylabel("Local Y (metres)")

    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()

    fig.savefig(
        output_dir / f"{partition}_{trip.trip_id}.png",
        dpi=160,
    )

    plt.close(fig)


# ---------------------------------------------------------
# 7. DATA QUALITY LOG
# ---------------------------------------------------------

def write_quality_log(audit, output_dir):
    """Write a short data-quality summary."""

    lines = [
        "PathSnap - Data Quality Log",
        "============================",
        "",
        f"Total trips: {len(audit)}",
        "",
    ]

    for partition, group in audit.groupby("part"):
        lines.extend([
            f"Partition: {partition}",
            f"Number of trips: {len(group)}",
            f"Total GPS points: {group['n_pts'].sum()}",
            f"Missing values: {group['nan_values'].sum()}",
            (
                "Median GPS points per trip: "
                f"{group['n_pts'].median():.1f}"
            ),
            (
                "Mean of per-trip median GPS-to-route distance: "
                f"{group['route_dist_median_m'].mean():.2f} metres"
            ),
            "",
        ])

    lines.extend([
        "Observations",
        "------------",
        (
            "Training trips 33 and 54 were identified by the "
            "project pipeline as defective and quarantined."
        ),
        "GPS time gaps are not always exactly one second.",
        "Student-test and training data must be analysed separately.",
        "The reference route is used for distance analysis only.",
        (
            "Time-gap and route-distance histograms display "
            "the central 99% of observations."
        ),
        (
            "Observations outside the displayed histogram range "
            "are excluded only from visualization, not from the dataset."
        ),
        "",
        "Note: These are development partitions, not evaluator data.",
    ])

    log_path = output_dir / "data_quality_log.txt"

    log_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ---------------------------------------------------------
# 8. MAIN FUNCTION
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate PathSnap EDA figures."
    )

    parser.add_argument("root")
    parser.add_argument("audit_csv")
    parser.add_argument("output_dir")

    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir)

    trajectory_dir = (
        output_dir / "gps_trajectories"
    )

    network_dir = (
        output_dir / "road_networks"
    )

    for directory in [
        output_dir,
        trajectory_dir,
        network_dir,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    audit = load_audit(args.audit_csv)

    print("Generating missing-value plot...")
    plot_missing_values(
        audit,
        output_dir,
    )

    print("Generating GPS time-gap histogram...")
    plot_time_gaps(
        root,
        output_dir,
    )

    print("Generating GPS-to-route distance histogram...")
    plot_route_distances(
        root,
        output_dir,
    )

    for partition in [
        "train",
        "student_test",
    ]:
        trips = list_trips(
            root,
            partition,
        )

        if not trips:
            print(
                f"No trips found in {partition}"
            )
            continue

        # Use the first trip as a representative example.
        trip_dir = trips[0]

        print(
            f"Generating trajectory: {partition}"
        )

        plot_trajectory(
            trip_dir,
            partition,
            trajectory_dir,
        )

        print(
            f"Generating road network: {partition}"
        )

        plot_road_network(
            trip_dir,
            partition,
            network_dir,
        )

    print("Generating data-quality log...")

    write_quality_log(
        audit,
        output_dir,
    )

    print("\nEDA generation completed.")

    print(
        f"Output directory: {output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
