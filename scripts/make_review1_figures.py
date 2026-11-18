"""Figures for Review 1, one or two per slide, drawn only from data and results in this repo.

    python scripts/make_review1_figures.py DATA_RAW DATA_PROCESSED PROBS_DIR RESULTS_DIR OUT_DIR

DATA_RAW        train/ and student_test/ trip folders (never the evaluator partition)
DATA_PROCESSED  output of build_dataset.py
PROBS_DIR       out-of-fold / student-test probabilities written by train_models.py
Colours: reference palette checked with the dataviz validator (5 slots, light mode).
Categorical slots are assigned to models in a fixed order and never reused.
"""
import json
import os
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shapely  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch  # noqa: E402
from shapely import STRtree  # noqa: E402
from shapely.geometry import LineString, box as sbox  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from pathsnap.candidates import build_candidates  # noqa: E402
from pathsnap.features import FEATURE_COLUMNS, build_features  # noqa: E402
from pathsnap.labels import align_labels  # noqa: E402
from pathsnap.segments import segment_ids  # noqa: E402
from pathsnap.trips import list_trips, load_trip  # noqa: E402

RAW, PROC, PROBS, RES, OUT = sys.argv[1:6]
os.makedirs(OUT, exist_ok=True)

SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
SEQ = LinearSegmentedColormap.from_list("blue", ["#f0f6fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])

ORDER = ["lr", "gnb", "rf", "hgb", "mlp"]                      # order on slides
NAME = dict(lr="Logistic\nRegression", gnb="Gaussian\nNaive Bayes", rf="Random\nForest",
            hgb="HistGradient\nBoosting", mlp="MLP")
FLAT = {k: v.replace("\n", " ") for k, v in NAME.items()}
COLOR = dict(hgb=BLUE, rf=ORANGE, mlp=AQUA, lr=YELLOW, gnb=MAGENTA)   # fixed slot per model

plt.rcParams.update({
    "font.family": "Segoe UI", "font.size": 13, "text.color": INK,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.linewidth": 1.0,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 12,
})


def csv(name):
    return pd.read_csv(os.path.join(RES, name), dtype={"trip": str})


audit, summ, runs, folds = csv("data_audit_per_trip.csv"), csv("dataset_build_summary.csv"), csv("model_runs.csv"), csv("cv_folds.csv")
stp, diag, conf, base = csv("student_test_per_trip.csv"), csv("model_diagnostics.csv"), csv("confusion_matrices.csv"), csv("baseline_student_test.csv")
runs = runs.set_index("model")
clean_ids = set(summ[~summ.quarantine].trip)
BASE_TOP1 = float((base.top1_with_heading * base.labelled_points).sum() / base.labelled_points.sum())
BASE_NEAREST = float((base.top1_nearest_only * base.labelled_points).sum() / base.labelled_points.sum())


def save(fig, name, header=None, sub=None, h=None):
    if header:
        fig.text(0.0, 1.0, header, fontsize=19, fontweight="bold", va="bottom", ha="left", transform=fig.transFigure)
    if sub:
        fig.text(0.0, 0.0, sub, fontsize=12, color=INK2, va="top", ha="left", transform=fig.transFigure)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    print("saved", name, flush=True)


def title_block(ax_or_fig, title, sub=None, fig=None):
    f = fig or ax_or_fig.figure
    w = f.get_figwidth()
    f.suptitle(textwrap.fill(title, int(w * 6.6)), x=0.0, y=1.0, ha="left", va="bottom", fontsize=19, fontweight="bold")
    if sub:
        f.text(0.0, -0.005, textwrap.fill(sub, int(w * 11.5)), ha="left", va="top", fontsize=11.5, color=INK2)


def clean_axes(ax, grid="y"):
    ax.grid(False)
    if grid:
        ax.grid(True, axis=grid)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ----------------------------------------------------------------- geometry helpers
def route_segments(trip, arcs):
    a = trip.arc_a[arcs]
    return np.stack([a, a + trip.arc_vec[arcs]], axis=1)


def nearest(tree, pts):
    idx, d = tree.query_nearest(pts, return_distance=True, all_matches=False)
    out = np.empty(len(pts))
    out[idx[0]] = d
    return out


def distances(trip):
    pts = shapely.points(trip.xy)
    seg_all = shapely.linestrings(np.stack([trip.arc_a, trip.arc_a + trip.arc_vec], axis=1))
    d_any = nearest(STRtree(seg_all), pts)
    d_route = nearest(STRtree(seg_all[trip.route]), pts)
    return d_route, d_any


def draw_network(ax, trip, box, lw=0.5, color=AXIS):
    x0, x1, y0, y1 = box
    a = trip.arc_a
    m = (a[:, 0] > x0) & (a[:, 0] < x1) & (a[:, 1] > y0) & (a[:, 1] < y1)
    segs = np.stack([a[m], a[m] + trip.arc_vec[m]], axis=1)
    ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, zorder=1))


def map_axes(ax, box):
    ax.set_xlim(box[0], box[1]); ax.set_ylim(box[2], box[3]); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def scale_bar(ax, box, meters, label):
    x0, x1, y0, y1 = box
    x = x0 + 0.04 * (x1 - x0); y = y0 + 0.04 * (y1 - y0)
    ax.plot([x, x + meters], [y, y], color=INK, lw=2.5, solid_capstyle="butt", zorder=9)
    ax.text(x + meters / 2, y + 0.015 * (y1 - y0), label, ha="center", va="bottom", fontsize=11, color=INK, zorder=9)


def trip_box(trip, pad=250):
    allp = np.vstack([trip.xy, trip.arc_a[trip.route]])
    return (allp[:, 0].min() - pad, allp[:, 0].max() + pad, allp[:, 1].min() - pad, allp[:, 1].max() + pad)


def pick_example_trip():
    m = summ[(summ.partition == "train") & ~summ.quarantine][["trip"]].merge(audit[["trip", "n_pts", "route_len_m", "route_dist_median_m"]], on="trip")
    c = m[(m.n_pts.between(900, 2200)) & (m.route_len_m.between(5000, 9000))].sort_values("route_dist_median_m")
    return c.iloc[0].trip


# ================================================================= slide 1 : the problem
def fig_trip_map():
    tid = pick_example_trip()
    trip = load_trip(os.path.join(RAW, "train", tid))
    d_route, _ = distances(trip)
    box = trip_box(trip, 200)
    # zoom on the window with the largest GPS-to-route offset (smoothed so one spike does not win)
    k = int(np.argmax(np.convolve(d_route, np.ones(15) / 15, mode="same")))
    cx, cy = trip.xy[k]
    zbox = (cx - 130, cx + 130, cy - 100, cy + 100)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.4), gridspec_kw=dict(width_ratios=[1.15, 1]))
    for ax, b, ps, rl in ((a1, box, 5, 2.2), (a2, zbox, 22, 3.2)):
        draw_network(ax, trip, b)
        ax.add_collection(LineCollection(route_segments(trip, trip.route), colors=AQUA, linewidths=rl, zorder=3))
        ax.scatter(trip.xy[:, 0], trip.xy[:, 1], s=ps, c=BLUE, zorder=4, linewidths=0)
        map_axes(ax, b)
    a1.add_patch(plt.Rectangle((zbox[0], zbox[2]), zbox[1] - zbox[0], zbox[3] - zbox[2], fill=False, ec=INK, lw=1.4, zorder=6))
    a1.text((zbox[0] + zbox[1]) / 2, zbox[3] + 25, "zoom", fontsize=11, color=INK, va="bottom", ha="center", zorder=6,
            bbox=dict(fc=SURF, ec="none", pad=1.5))
    km = (box[1] - box[0]) / 5
    sb = 1000.0 if km > 1000 else 500.0
    scale_bar(a1, box, sb, f"{int(sb)} m")
    scale_bar(a2, zbox, 50.0, "50 m")
    a1.set_title(f"Whole trip ({len(trip.t):,} GPS fixes)", loc="left", fontsize=13, color=INK2)
    a2.set_title("Zoom: the fixes wander off the road", loc="left", fontsize=13, color=INK2)
    h = [plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=8, label="GPS fixes"),
         plt.Line2D([], [], color=AQUA, lw=3, label="Reference route"),
         plt.Line2D([], [], color=AXIS, lw=2, label="Road network")]
    a1.legend(handles=h, loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=12, frameon=True, facecolor=SURF, edgecolor="none", framealpha=0.92)
    title_block(fig, "GPS fixes drift off the road, so we have to snap them back",
                f"Train trip {tid}. Blue: raw GPS. Green: the reference route. Grey: the road network supplied for this trip.")
    save(fig, "01_trip_map.png")
    return tid


# ================================================================= slide 2 : data
def fig_world():
    a = audit.copy()
    fig, ax = plt.subplots(figsize=(11, 5.6))
    for part, col, lab in (("train", BLUE, "train (80 trips)"), ("student_test", ORANGE, "student-test (10 trips)")):
        s = a[a.part == part]
        ax.scatter(s.lon_mean, s.lat_mean, s=90 if part == "student_test" else 70, c=col, linewidths=2,
                   edgecolors=SURF, zorder=3, label=lab)
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 85)
    ax.set_xticks(range(-180, 181, 60)); ax.set_yticks(range(-60, 81, 30))
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.grid(True, axis="both")
    ax.legend(loc="lower left", ncol=2)
    title_block(fig, "The 90 trips come from all over the world", "Each dot is one trip (mean of its GPS fixes). "
                "No two cities share a road network, so a road's id means nothing across trips.")
    save(fig, "02_trips_world.png")


def fig_ladder():
    labels = ["Nearest road only", "Nearest road + heading rule", "HistGradientBoosting\n(our classifier)", "Boosting + HMM\n(Review 2 preview)"]
    vals = [BASE_NEAREST, BASE_TOP1, float(runs.loc["hgb", "student_top1"]), 0.9315]
    cols = [AXIS, AXIS, BLUE, "#9ec5f4"]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    y = np.arange(len(vals))[::-1]
    ax.barh(y, [v * 100 for v in vals], height=0.52, color=cols, edgecolor=SURF, linewidth=2)
    for yi, v in zip(y, vals):
        ax.text(v * 100 + 1, yi, f"{v*100:.1f}%", va="center", fontsize=15, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=13, color=INK)
    ax.set_xlim(0, 105); ax.set_xlabel("Student-test Top-1 accuracy (%)")
    clean_axes(ax, "x")
    ax.tick_params(axis="y", length=0)
    title_block(fig, "Direction is the whole game: the nearest road is right only about half the time",
                "Nearly every road has an opposite-direction twin at the same distance, so distance cannot tell them apart. "
                "Student-test, 10 trips, point-weighted.")
    save(fig, "03_accuracy_ladder.png")


def fig_sampling():
    a = audit.copy()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw=dict(width_ratios=[1, 1.15]))
    bins = np.arange(0.50, 1.0001, 0.025)
    a1.hist(a.dt_eq1, bins=bins, color=BLUE, edgecolor=SURF, linewidth=2)
    irregular = int((a.dt_eq1 < 0.9).sum())
    a1.set_xlabel("share of steps that are exactly 1 second"); a1.set_ylabel("number of trips")
    clean_axes(a1, "y")
    a1.set_title(f"{irregular} of 90 trips are not really 1 Hz", loc="left", fontsize=13.5, color=INK)
    a1.text(0.04, 0.50, f"{irregular} trips below 90%", transform=a1.transAxes, fontsize=12.5, color=INK2)
    s = a.sort_values("dt_max").reset_index(drop=True)
    a2.scatter(s.index + 1, s.dt_max, s=36, c=BLUE, linewidths=1.5, edgecolors=SURF, zorder=3)
    a2.set_yscale("log"); a2.axhline(60, color=INK, lw=1.2, zorder=2)
    a2.text(2, 70, "60 s: the HMM chain restarts above this", fontsize=11.5, color=INK, va="bottom")
    for _, r in s.tail(2).iterrows():
        a2.annotate(f"{int(r.dt_max):,} s", (_ + 1, r.dt_max), xytext=(-8, 4), textcoords="offset points", ha="right", fontsize=11, color=INK2)
    a2.set_xlabel("trips, sorted by largest time gap"); a2.set_ylabel("largest time gap in the trip (s, log)")
    clean_axes(a2, "y")
    a2.set_title("Some trips have gaps of minutes, hours or days", loc="left", fontsize=13.5, color=INK)
    title_block(fig, "Timing is irregular, so speed and the HMM must use the real time between fixes",
                "All 90 trips (train and student-test).")
    save(fig, "04_sampling_gaps.png")


def fig_noise():
    d = []
    for t in sorted(clean_ids):
        part = "student_test" if t in set(summ[summ.partition == "student_test"].trip) else "train"
        trip = load_trip(os.path.join(RAW, part, t))
        seg = segment_ids(trip.t, trip.xy)
        lab, _ = align_labels(trip, seg)
        d.append(lab.label_dist.dropna().to_numpy())
    d = np.concatenate(d)
    d.sort()
    ecdf = np.arange(1, len(d) + 1) / len(d)
    q = {p: float(np.quantile(d, p)) for p in (0.5, 0.95, 0.99)}
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.plot(d, ecdf * 100, color=BLUE, lw=2.5, zorder=3)
    ax.set_xlim(0, 60); ax.set_ylim(0, 101)
    for p, dy in ((0.5, 12), (0.95, 8)):
        ax.plot([q[p]], [p * 100], "o", ms=9, color=BLUE, mec=SURF, mew=2, zorder=4)
        ax.annotate(f"{int(p*100)}% of fixes are within {q[p]:.1f} m", (q[p], p * 100), xytext=(q[p] + 4, p * 100 - dy),
                    fontsize=13, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    ax.axvline(40, color=INK, lw=1.2, zorder=2)
    ax.text(40.8, 8, "candidate radius: 40 m", rotation=90, fontsize=12, color=INK, va="bottom")
    ax.set_xlabel("distance from a GPS fix to its reference road (m)"); ax.set_ylabel("share of fixes (%)")
    clean_axes(ax, "y")
    title_block(fig, "A GPS fix is usually within a few metres of the road, and 40 m catches nearly all of them",
                f"{len(d):,} labelled fixes on the 88 clean train and student-test trips.")
    save(fig, "05_gps_noise.png")


def fig_defects(clean_tid):
    cases = [(clean_tid, "train", "Clean trip " + clean_tid), ("00000033", "train", "Trip 33: route skips half the drive"),
             ("00000054", "train", "Trip 54: roads missing from the network")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2), gridspec_kw=dict(height_ratios=[1.25, 1]))
    ymax = 0
    series = []
    for tid, part, _ in cases:
        trip = load_trip(os.path.join(RAW, part, tid))
        dr, da = distances(trip)
        series.append((trip, dr, da))
    for j, ((trip, dr, da), (tid, _, ttl)) in enumerate(zip(series, cases)):
        ax = axes[0, j]
        box = trip_box(trip, 250)
        draw_network(ax, trip, box)
        ax.add_collection(LineCollection(route_segments(trip, trip.route), colors=AQUA, linewidths=2.2, zorder=3))
        ax.scatter(trip.xy[:, 0], trip.xy[:, 1], s=4, c=BLUE, zorder=4, linewidths=0)
        map_axes(ax, box)
        ax.set_title(ttl, loc="left", fontsize=13, color=INK)
        a2 = axes[1, j]
        n = len(dr); b = np.array_split(np.arange(n), 40)
        x = np.array([np.mean(i) / n * 100 for i in b])
        a2.plot(x, [np.median(da[i]) for i in b], color=BLUE, lw=2.2, label="to nearest road of any kind")
        a2.plot(x, [np.median(dr[i]) for i in b], color=ORANGE, lw=2.2, label="to the reference route")
        a2.set_xlabel("progress along the trip (%)"); clean_axes(a2, "y")
        a2.set_ylim(0, 230)
        if j == 0:
            a2.set_ylabel("median distance (m)")
            a2.legend(loc="upper left", fontsize=11.5)
    h = [plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=7, label="GPS fixes"),
         plt.Line2D([], [], color=AQUA, lw=3, label="Reference route")]
    axes[0, 2].legend(handles=h, loc="upper right", fontsize=11.5, frameon=True, facecolor=SURF, edgecolor="none", framealpha=0.92)
    title_block(fig, "Two training trips have broken references, so we quarantine them",
                "Trip 33: the fixes sit on real roads (blue line near zero) but far from the reference route. "
                "Trip 54: the fixes are far from every road in the supplied network. Together they are 12% of training points.")
    fig.subplots_adjust(hspace=0.28, wspace=0.18)
    save(fig, "06_defective_trips.png")


# ================================================================= slide 3 : problem framing
def fig_candidates():
    tid = "00000040"
    trip = load_trip(os.path.join(RAW, "student_test", tid))
    seg = segment_ids(trip.t, trip.xy)
    lab, _ = align_labels(trip, seg)
    cands = build_candidates(trip)
    feats = build_features(trip, cands, seg, lab.label_arc.to_numpy())
    best = None
    for p, g in feats.groupby("point_idx"):
        ref = lab.label_arc.iloc[p]
        if ref < 0 or not g.head_valid.iloc[0] or g.speed_mps.iloc[0] < 4 or not (7 <= len(g) <= 11):
            continue
        twin = trip.reverse_arc[ref]
        ids = set(g.arc_id)
        if twin >= 0 and twin in ids and g[g.arc_id == ref].dist_m.iloc[0] < 10:
            if best is None or len(g) > best[1]:
                best = (p, len(g), ref, twin)
    p, _, ref, twin = best
    g = feats[feats.point_idx == p]
    P = trip.xy[p]
    hx = g.cos_head.iloc[0]
    # motion direction from the reference arc's neighbours
    lo, hi = max(p - 3, 0), min(p + 3, trip.n_points - 1)
    mv = trip.xy[hi] - trip.xy[lo]; mv = mv / np.linalg.norm(mv)
    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    ax.add_patch(Circle(P, 40, fill=False, ec=AXIS, lw=1.5, zorder=1))
    ax.text(P[0] + 28, P[1] + 31, "40 m", color=MUTED, fontsize=11, ha="left")
    view = sbox(P[0] - 62, P[1] - 52, P[0] + 62, P[1] + 52)
    print("candidate example: ref", int(ref), "twin", int(twin), "arcs", sorted(int(v) for v in g.arc_id))
    for _, r in g.iterrows():
        a = int(r.arc_id)
        A, B = trip.arc_a[a], trip.arc_a[a] + trip.arc_vec[a]
        d = trip.arc_dir[a]; off = np.array([d[1], -d[0]]) * 1.6
        col, lw, z = (BLUE, 4.2, 5) if a == ref else (ORANGE, 4.2, 5) if a == twin else (MUTED, 2.0, 3)
        cut = LineString([A + off, B + off]).intersection(view)
        if cut.is_empty or cut.geom_type != "LineString":
            continue
        c0, c1 = np.array(cut.coords[0]), np.array(cut.coords[-1])
        ax.plot([c0[0], c1[0]], [c0[1], c1[1]], color=col, lw=lw, solid_capstyle="butt", zorder=z)
        mid = (c0 + c1) / 2; u = (c1 - c0) / max(np.linalg.norm(c1 - c0), 1e-9)
        ax.annotate("", xy=mid + u * 3, xytext=mid - u * 3, arrowprops=dict(arrowstyle="-|>", color=col, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=20), zorder=z + 1)
    ax.plot(*P, "o", ms=13, color=INK, mec=SURF, mew=3, zorder=8)
    ax.annotate("", xy=P + mv * 26, xytext=P, arrowprops=dict(arrowstyle="-|>", color=INK, lw=2, mutation_scale=16), zorder=9)
    ax.text(P[0] + mv[0] * 30, P[1] + mv[1] * 30 - 3, "travel direction", fontsize=12, color=INK, va="top", ha="center", zorder=9,
            bbox=dict(fc=SURF, ec="none", pad=1.5))
    ax.set_xlim(P[0] - 62, P[0] + 62); ax.set_ylim(P[1] - 52, P[1] + 52); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    h = [plt.Line2D([], [], color=INK, marker="o", ls="", ms=10, label="GPS fix"),
         plt.Line2D([], [], color=BLUE, lw=4, label="reference road (label = 1)"),
         plt.Line2D([], [], color=ORANGE, lw=4, label="same road, opposite direction (twin)"),
         plt.Line2D([], [], color=MUTED, lw=2.5, label=f"other candidates ({len(g) - 2})")]
    ax.legend(handles=h, loc="center left", fontsize=12, bbox_to_anchor=(1.02, 0.5))
    title_block(fig, f"For every GPS fix we score about {len(g)} candidate roads, and always include the opposite direction",
                f"Real fix from student-test trip {tid}. Arrows show each road's direction of travel; the twin is drawn beside its road. "
                "The classifier answers one question per candidate: is this the reference road?")
    save(fig, "07_candidates_example.png")


def box(ax, x, y, w, h, text, sub, fill, ec, lw=1.6, ls="-", tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.12", fc=fill, ec=ec, lw=lw, ls=ls, zorder=2))
    ax.text(x + w / 2, y + h * 0.66, text, ha="center", va="center", fontsize=14, fontweight="bold", color=tc, zorder=3)
    ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=10.5, color=INK2, zorder=3, linespacing=1.35)


def fig_pipeline():
    n_rows = int(summ.n_cand_rows.sum())
    fig, ax = plt.subplots(figsize=(16.5, 4.4))
    ax.set_xlim(0, 16.5); ax.set_ylim(0, 4.2); ax.axis("off")
    stages = [
        ("Inputs", "GPS track\n+ road network\nper trip", "#fcfcfb", AXIS, "-"),
        ("1  Candidates", f"10 nearest roads\nwithin 40 m + twin\n{n_rows/1e6:.2f} M rows", "#e8f1fc", BLUE, "-"),
        ("2  Features", f"{len(FEATURE_COLUMNS)} per candidate:\ndistance, side, heading,\nspeed, rank ...", "#e8f1fc", BLUE, "-"),
        ("3  Classifier", "is this the\nreference road?\n5 models compared", "#e8f1fc", BLUE, "-"),
        ("4  HMM + Viterbi", "connect the choices\nalong the network\n(Review 2)", SURF, MUTED, "--"),
        ("Route", "one connected\nroad sequence", "#fcfcfb", AXIS, "-"),
    ]
    w, gap, x = 2.3, 0.45, 0.15
    for i, (t, s, fill, ec, ls) in enumerate(stages):
        box(ax, x, 1.0, w, 2.0, t, s, fill, ec, ls=ls, lw=2.0 if ec == BLUE else 1.5)
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + w + gap - 0.05, 2.0), xytext=(x + w + 0.05, 2.0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=2, mutation_scale=16))
        x += w + gap
    ax.text(0.15 + 1 * (w + gap), 3.35, "Today (Review 1)", fontsize=12, color=BLUE, fontweight="bold")
    ax.plot([0.15 + (w + gap), 0.15 + 4 * (w + gap) - gap], [3.2, 3.2], color=BLUE, lw=2)
    ax.text(0.15 + 4 * (w + gap), 3.35, "Review 2", fontsize=12, color=MUTED, fontweight="bold")
    ax.plot([0.15 + 4 * (w + gap), 0.15 + 4 * (w + gap) + w], [3.2, 3.2], color=MUTED, lw=2)
    save(fig, "08_pipeline.png")


# ================================================================= slide 4 : evaluation
def fig_partitions():
    f = folds[folds.model == "hgb"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6.6), gridspec_kw=dict(height_ratios=[1, 1.35], hspace=0.7))
    parts = [("clean train: 78 trips", 78, BLUE, "#ffffff"), ("", 2, MUTED, "#ffffff"), ("", 10, ORANGE, "#ffffff"), ("", 10, "#e1e0d9", INK)]
    x = 0
    for lab, n, col, tc in parts:
        a1.barh(0, n, left=x, height=0.55, color=col, edgecolor=SURF, linewidth=3)
        if lab:
            a1.text(x + n / 2, 0, lab, ha="center", va="center", color=tc, fontsize=12.5, fontweight="bold")
        x += n
    a1.annotate("2 quarantined\n(trips 33, 54)", (79, 0.28), xytext=(76, 0.95), ha="center", fontsize=11.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED))
    a1.annotate("student-test: 10 trips\nreporting only", (89, -0.28), xytext=(87, -0.95), ha="center", va="top", fontsize=11.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED))
    a1.annotate("evaluator: 10 trips\nnever touched", (99, 0.28), xytext=(99, 0.95), ha="center", fontsize=11.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED))
    a1.set_xlim(0, 104); a1.set_ylim(-1.3, 1.5); a1.axis("off")
    a1.set_title("100 trips split three ways. We fit only on the 78 clean train trips.", loc="left", fontsize=13.5, color=INK, pad=12)
    ys = np.arange(len(f))[::-1]
    a2.barh(ys, f.n_val_trips, height=0.55, color=BLUE, edgecolor=SURF, linewidth=3)
    for y, (_, r) in zip(ys, f.iterrows()):
        a2.text(r.n_val_trips + 0.4, y, f"{int(r.n_val_trips)} trips, {int(r.n_val_points):,} points held out", va="center", fontsize=12, color=INK)
    a2.set_yticks(ys); a2.set_yticklabels([f"fold {int(k) + 1}" for k in f.fold]); a2.set_xlim(0, 32)
    a2.set_xlabel("number of held-out trips"); clean_axes(a2, "x"); a2.tick_params(axis="y", length=0)
    a2.set_title("5-fold cross-validation, grouped by trip: a trip is never split across folds", loc="left", fontsize=13.5, color=INK, pad=12)
    title_block(fig, "How we keep the evaluation honest", "The three Moscow trips share roads, so they always sit in the same fold.")
    save(fig, "09_partitions_and_cv.png")


def fig_points_per_trip():
    a = audit
    fig, ax = plt.subplots(figsize=(11, 5.2))
    bins = np.logspace(np.log10(250), np.log10(20000), 22)
    ax.hist(a.n_pts, bins=bins, color=BLUE, edgecolor=SURF, linewidth=2)
    ax.set_xscale("log")
    ax.axvline(1000, color=INK, lw=1.4, zorder=3)
    ax.text(930, ax.get_ylim()[1] * 0.97, "cap: 1,000 points\nper trip when fitting", fontsize=12, color=INK, va="top", ha="right")
    ax.annotate("trip 54: 19,313", (19313, 0.4), xytext=(9000, 8), fontsize=12, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.set_xlabel("GPS fixes in the trip (log scale)"); ax.set_ylabel("number of trips")
    clean_axes(ax, "y")
    ax.set_xticks([300, 1000, 3000, 10000]); ax.set_xticklabels(["300", "1,000", "3,000", "10,000"])
    title_block(fig, "Trip sizes differ by 74 times, so we cap points per trip", "All 90 trips. Without the cap one trip would dominate training.")
    save(fig, "10_points_per_trip.png")


# ================================================================= slide 6 : accuracy
def fig_cv():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)
    y = np.arange(len(ORDER))[::-1]
    for ax, col, ttl in ((a1, "top1", "Top-1 per fold"), (a2, "macro_f1", "Macro F1 per fold")):
        for yi, m in zip(y, ORDER):
            v = folds[folds.model == m][col].to_numpy()
            ax.scatter(v, np.full(len(v), yi), s=70, c=COLOR[m], linewidths=1.5, edgecolors=SURF, alpha=0.85, zorder=3)
            ax.plot([v.mean()] * 2, [yi - 0.28, yi + 0.28], color=INK, lw=2.6, zorder=4, solid_capstyle="butt")
            ax.text(max(v.max(), v.mean()) + 0.004, yi, f"{v.mean():.3f}", va="center", fontsize=12, fontweight="bold" if m == "hgb" else "normal", color=INK)
        ax.set_yticks(y); ax.set_yticklabels([FLAT[m] for m in ORDER], color=INK, fontsize=12.5); ax.tick_params(axis="y", length=0)
        ax.set_title(ttl, loc="left", fontsize=13.5, color=INK); clean_axes(ax, "x")
    a1.set_xlim(0.82, 0.965); a2.set_xlim(0.79, 0.985); a1.set_ylim(-1.2, 4.5)
    a1.set_xlabel("observation-level Top-1"); a2.set_xlabel("macro F1 (candidate rows)")
    h = [plt.Line2D([], [], marker="o", ls="", color=MUTED, ms=8, label="one fold"), plt.Line2D([], [], color=INK, lw=3, label="mean of 5 folds")]
    a1.legend(handles=h, loc="lower left")
    title_block(fig, "Forest, boosting and MLP are tied; logistic regression trails; naive Bayes is clearly worse",
                "5 trip-grouped folds on 78 clean train trips. Fold-to-fold spread (about 0.014 macro F1) is bigger than the gaps among the top three.")
    save(fig, "11_cv_by_fold.png")


def fig_student_trips():
    fig, ax = plt.subplots(figsize=(12, 5.8))
    y = np.arange(len(ORDER))[::-1]
    for yi, m in zip(y, ORDER):
        s = stp[stp.model == m]
        ax.scatter(s.top1, np.full(len(s), yi), s=60, c=COLOR[m], linewidths=1.5, edgecolors=SURF, alpha=0.85, zorder=3)
        mean = float(runs.loc[m, "student_top1"])
        ax.plot([mean] * 2, [yi - 0.3, yi + 0.3], color=INK, lw=2.8, zorder=4, solid_capstyle="butt")
        ax.text(1.005, yi, f"{mean*100:.1f}%", va="center", fontsize=13, fontweight="bold" if m == "hgb" else "normal", color=INK)
    ax.axvline(BASE_TOP1, color=INK, lw=1.3, zorder=2)
    ax.text(BASE_TOP1 - 0.004, y[0] + 0.55, f"rule baseline {BASE_TOP1*100:.1f}%", ha="right", fontsize=12, color=INK)
    low = stp[(stp.model == "hgb")].sort_values("top1").iloc[0]
    ax.annotate(f"trip {int(low.trip)}", (low.top1, y[ORDER.index('hgb')]), xytext=(low.top1 - 0.03, y[ORDER.index('hgb')] + 0.42), fontsize=11.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.set_yticks(y); ax.set_yticklabels([FLAT[m] for m in ORDER], color=INK, fontsize=12.5); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.70, 1.03); ax.set_ylim(-1.3, 4.9); ax.set_xlabel("Top-1 accuracy on one student-test trip")
    clean_axes(ax, "x")
    h = [plt.Line2D([], [], marker="o", ls="", color=MUTED, ms=8, label="one trip (10 per model)"), plt.Line2D([], [], color=INK, lw=3, label="all points pooled")]
    ax.legend(handles=h, loc="lower left")
    title_block(fig, "On unseen trips, boosting leads at 89.1%, and every strong model beats the rule", "Student-test, 10 trips, never used for fitting. Trip 77 is hard for every model.")
    save(fig, "12_student_test_by_trip.png")


def fig_time_acc():
    fig, ax = plt.subplots(figsize=(11, 5.8))
    pos = dict(lr=(8, 0), gnb=(8, 0), rf=(10, 0), hgb=(10, 0), mlp=(10, 0))
    for m in ORDER:
        x, y = float(runs.loc[m, "final_fit_seconds"]), float(runs.loc[m, "student_top1"]) * 100
        ax.scatter(x, y, s=170, c=COLOR[m], linewidths=2.5, edgecolors=SURF, zorder=4)
        dx, dy = {"lr": (10, 10), "gnb": (10, -4), "rf": (-10, 10), "hgb": (10, 10), "mlp": (-10, 10)}[m]
        ax.annotate(f"{FLAT[m]}\n{x:g} s", (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=12.5, color=INK,
                    ha="left" if dx > 0 else "right", va="bottom" if dy > 0 else "top", fontweight="bold" if m == "hgb" else "normal")
    ax.axhline(BASE_TOP1 * 100, color=INK, lw=1.2, zorder=2)
    ax.text(0.2, BASE_TOP1 * 100 + 0.15, "rule baseline 86.0%", fontsize=11.5, color=INK)
    ax.set_xscale("log"); ax.set_xlim(0.15, 400); ax.set_ylim(82, 90.5)
    ax.set_xlabel("time to fit the final model (s, log scale)"); ax.set_ylabel("student-test Top-1 (%)")
    ax.grid(True, axis="both")
    title_block(fig, "Boosting matches the best accuracy in 2.4 s; the forest and MLP take 20 to 45 times longer",
                "Final fit on 78 train trips (about 457,000 candidate rows), one machine. Cheap training makes the Bayesian search in Review 2 affordable.")
    save(fig, "13_time_vs_accuracy.png")


# ================================================================= slide 7 : what the numbers mean
def fig_overfit():
    d = diag.set_index("model")
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    y = np.arange(len(ORDER))[::-1]
    for yi, m in zip(y, ORDER):
        tr, st = d.loc[m, "train_top1"] * 100, d.loc[m, "student_top1"] * 100
        ax.plot([st, tr], [yi, yi], color=AXIS, lw=3, zorder=2, solid_capstyle="round")
        ax.scatter([tr], [yi], s=150, c=ORANGE, linewidths=2.5, edgecolors=SURF, zorder=4)
        ax.scatter([st], [yi], s=150, c=BLUE, linewidths=2.5, edgecolors=SURF, zorder=4)
        ax.text(max(tr, st) + 0.5, yi, f"gap {tr - st:.1f} pts", va="center", fontsize=12.5, color=INK, fontweight="bold" if m == "rf" else "normal")
    ax.set_yticks(y); ax.set_yticklabels([FLAT[m] for m in ORDER], color=INK, fontsize=12.5); ax.tick_params(axis="y", length=0)
    ax.set_xlim(82, 99); ax.set_xlabel("Top-1 accuracy (%)"); clean_axes(ax, "x")
    h = [plt.Line2D([], [], marker="o", ls="", color=ORANGE, ms=10, label="training rows"), plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=10, label="student-test (unseen trips)")]
    ax.legend(handles=h, loc="lower right")
    title_block(fig, "The random forest overfits; logistic regression underfits", "A big train-to-test gap means memorising (forest). Low on both means too simple (logistic regression, naive Bayes).")
    save(fig, "14_overfitting.png")


def load_student(model):
    frames = []
    for t in sorted(os.listdir(os.path.join(PROBS, model, "student_test"))):
        t = t.replace(".parquet", "")
        f = pd.read_parquet(os.path.join(PROC, "student_test", t + ".parquet"), columns=["trip_id", "point_idx", "arc_id", "label"])
        p = pd.read_parquet(os.path.join(PROBS, model, "student_test", t + ".parquet"))
        assert np.array_equal(f.arc_id.to_numpy(), p.sort_values("point_idx", kind="stable").arc_id.to_numpy()) or len(f) == len(p)
        f["prob"] = p.prob.to_numpy()
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def fig_calibration():
    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1.6, zorder=1)
    ax.text(0.03, 0.47, "grey diagonal = perfect calibration", color=INK2, fontsize=11.5, ha="left", va="center")
    bins = np.linspace(0, 1, 11)
    gnb_hi = None
    for m in ORDER:
        d = load_student(m); d = d[d.label >= 0]
        idx = np.clip(np.digitize(d.prob, bins) - 1, 0, 9)
        g = d.groupby(idx).agg(p=("prob", "mean"), o=("label", "mean"), n=("label", "size"))
        g = g[g.n >= 50]
        ax.plot(g.p, g.o, "-o", color=COLOR[m], lw=2.6, ms=8, mec=SURF, mew=1.8, zorder=3, label=FLAT[m])
        if m == "gnb":
            hi = g[g.p > 0.9]
            gnb_hi = (float(hi.p.iloc[-1]), float(hi.o.iloc[-1])) if len(hi) else None
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect("equal")
    ax.set_xlabel("predicted probability of being the reference road"); ax.set_ylabel("how often it really is")
    ax.grid(True, axis="both")
    ax.legend(loc="upper left", fontsize=12)
    if gnb_hi:
        ax.annotate(f"naive Bayes says {gnb_hi[0]*100:.0f}%,\nright {gnb_hi[1]*100:.0f}% of the time", gnb_hi, xytext=(0.36, 0.17), fontsize=12.5, va="bottom", color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED))
    title_block(fig, "Naive Bayes is over-confident; the other models are well calibrated",
                "Student-test, all candidate rows on labelled points. Calibration matters because the HMM uses these probabilities.")
    save(fig, "15_calibration.png")


def fig_errors():
    cats = dict(ok=0, twin=0, other=0, missing=0)
    for t in sorted(os.listdir(os.path.join(PROBS, "hgb", "student_test"))):
        t = t.replace(".parquet", "")
        trip = load_trip(os.path.join(RAW, "student_test", t), with_route=False)
        f = pd.read_parquet(os.path.join(PROC, "student_test", t + ".parquet"), columns=["point_idx", "arc_id", "label"])
        p = pd.read_parquet(os.path.join(PROBS, "hgb", "student_test", t + ".parquet"))
        f["prob"] = p.prob.to_numpy()
        for _, g in f[f.label >= 0].groupby("point_idx"):
            ref = g[g.label == 1].arc_id
            pred = int(g.arc_id.iloc[int(np.argmax(g.prob.to_numpy()))])
            if len(ref) == 0:
                cats["missing"] += 1
            elif pred == int(ref.iloc[0]):
                cats["ok"] += 1
            elif trip.reverse_arc[int(ref.iloc[0])] == pred:
                cats["twin"] += 1
            else:
                cats["other"] += 1
    total = sum(cats.values()); wrong = total - cats["ok"]
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    rows = [("Opposite direction of the right road", cats["twin"], ORANGE), ("A different road segment", cats["other"], BLUE),
            ("Right road missing from the candidates", cats["missing"], MUTED)]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, n, col) in zip(y, rows):
        ax.barh(yi, n, height=0.5, color=col, edgecolor=SURF, linewidth=2)
        ax.text(n + wrong * 0.012, yi, f"{n:,}  ({n / wrong * 100:.0f}% of mistakes)", va="center", fontsize=13, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=13, color=INK); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, wrong * 1.0); ax.set_xlabel("student-test fixes the classifier gets wrong"); clean_axes(ax, "x")
    title_block(fig, f"Boosting is right on {cats['ok'] / total * 100:.1f}% of fixes; about a third of its mistakes are a direction flip",
                f"{total:,} labelled student-test fixes, {wrong:,} wrong. The HMM adds memory of where the car just was. "
                "What the 'different segment' errors consist of is not analysed yet.")
    save(fig, "16_error_breakdown.png")
    return cats


def fig_curves():
    h = json.load(open(os.path.join(RES, "history_hgb.json"))); m = json.load(open(os.path.join(RES, "history_mlp.json")))
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.8))
    it = np.arange(len(h["train_score"]))
    a1.plot(it, -np.array(h["train_score"]), color=ORANGE, lw=2.4, label="training rows")
    a1.plot(it, -np.array(h["val_score"]), color=BLUE, lw=2.4, label="hold-out rows")
    a1.set_xlabel("boosting iteration"); a1.set_ylabel("log loss"); a1.set_title("Boosting: loss", loc="left", fontsize=13.5, color=INK); a1.legend()
    ep = np.arange(len(m["train_loss"]))
    a2.plot(ep, m["train_loss"], color=ORANGE, lw=2.4)
    a2.set_xlabel("epoch"); a2.set_ylabel("training log loss"); a2.set_title("MLP: training loss", loc="left", fontsize=13.5, color=INK)
    a3.plot(np.arange(len(m["val_score"])), np.array(m["val_score"]) * 100, color=BLUE, lw=2.4)
    a3.set_xlabel("epoch"); a3.set_ylabel("hold-out accuracy (%)"); a3.set_title("MLP: hold-out accuracy", loc="left", fontsize=13.5, color=INK)
    for a in (a1, a2, a3):
        clean_axes(a, "y")
    title_block(fig, "Training curves exist only for models that train in steps",
                "Boosting and the MLP stop early when the hold-out stops improving. The hold-out uses random rows, not whole trips, so it is for early stopping only. "
                "The forest, logistic regression and naive Bayes have no epochs, so no curve is drawn for them.")
    save(fig, "17_training_curves.png")


def fig_confusion():
    fig, axes = plt.subplots(1, 5, figsize=(17, 4.0))
    for ax, m in zip(axes, ORDER):
        r = conf[conf.model == m].iloc[0]
        cm = np.array([[r.tn, r.fp], [r.fn, r.tp]], dtype=float)
        pct = cm / cm.sum(1, keepdims=True)
        ax.imshow(pct, cmap=SEQ, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(cm[i, j]):,}\n{pct[i, j]*100:.1f}%", ha="center", va="center", fontsize=12,
                        color="#ffffff" if pct[i, j] > 0.5 else INK, fontweight="bold" if i != j else "normal")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred no", "pred yes"]); ax.set_yticks([0, 1])
        ax.set_yticklabels(["true no", "true yes"] if m == ORDER[0] else ["", ""])
        ax.set_title(FLAT[m], fontsize=13, color=INK, pad=8); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    title_block(fig, "Naive Bayes makes about five times as many false alarms as the other models",
                "Student-test candidate rows at threshold 0.5. Each cell: count and share of its true class.")
    save(fig, "18_confusion_matrices.png")


# ================================================================= run
if __name__ == "__main__":
    tid = fig_trip_map()
    fig_world(); fig_ladder(); fig_sampling(); fig_noise(); fig_defects(tid)
    fig_candidates(); fig_pipeline()
    fig_partitions(); fig_points_per_trip()
    fig_cv(); fig_student_trips(); fig_time_acc()
    fig_overfit(); fig_calibration(); cats = fig_errors(); fig_curves(); fig_confusion()
    print("error categories (boosting, student-test):", cats)
