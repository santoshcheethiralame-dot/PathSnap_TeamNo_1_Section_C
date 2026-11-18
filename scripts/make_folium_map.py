"""Folium map of one trip: GPS fixes, road geometry, reference route, predicted route.

    python scripts/make_folium_map.py DATA_RAW PARTITION TRIP_ID OUT_HTML

Layers (toggle with the layer control): GPS fixes, wrongly matched fixes, road
network, reference route, predicted route (classifier + HMM, same code path as
the API). The reference route is read here for the report only; the API never
sends it to the UI. Student-test trips for the report: pass PARTITION student_test.
"""
import os
import sys

import folium
import numpy as np
from branca.element import Element

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from pathsnap.labels import align_labels  # noqa: E402
from pathsnap.predict import predict_trip  # noqa: E402
from pathsnap.segments import segment_ids  # noqa: E402
from pathsnap.trips import PARTITIONS, load_trip  # noqa: E402

GREEN, BLUE, ORANGE, RED, GREY = "#1baf7a", "#2a78d6", "#eda100", "#e34948", "#898781"


def latlon(lonlat):
    return [[float(b), float(a)] for a, b in lonlat]


def lines_of(trip, arcs, lonlat_nodes):
    return [[[lonlat_nodes[trip.u[a]][1], lonlat_nodes[trip.u[a]][0]], [lonlat_nodes[trip.v[a]][1], lonlat_nodes[trip.v[a]][0]]] for a in arcs]


def build(raw, partition, tid, out):
    if partition not in PARTITIONS:
        raise SystemExit(f"partition must be one of {PARTITIONS}")
    trip_dir = os.path.join(raw, partition, tid)
    trip = load_trip(trip_dir)
    pred = predict_trip(trip_dir)
    seg = segment_ids(trip.t, trip.xy)
    lab, _ = align_labels(trip, seg)
    ref_arc, pred_arc = lab.label_arc.to_numpy(), np.array(pred["point_arc"])
    wrong = (ref_arc >= 0) & (pred_arc != ref_arc)
    n_lab = int((ref_arc >= 0).sum())

    nodes_ll = trip.to_lonlat(trip.node_xy)
    fixes = np.array(pred["fix_lonlat"])
    centre = [float(fixes[:, 1].mean()), float(fixes[:, 0].mean())]
    m = folium.Map(location=centre, zoom_start=15, tiles="OpenStreetMap", control_scale=True)

    # road geometry near the trip, one line per road (not per direction)
    lo, hi = trip.xy.min(0) - 100, trip.xy.max(0) + 100
    mid = trip.arc_a + trip.arc_vec / 2
    near = np.flatnonzero((mid[:, 0] > lo[0]) & (mid[:, 0] < hi[0]) & (mid[:, 1] > lo[1]) & (mid[:, 1] < hi[1]))
    twin = trip.reverse_arc[near]
    near = near[(twin < 0) | (near < twin)]
    roads = folium.FeatureGroup("Road network", show=True)
    folium.PolyLine(lines_of(trip, near, nodes_ll), color=GREY, weight=1.5, opacity=0.7).add_to(roads)   # one multi-line, keeps the file small
    roads.add_to(m)

    ref = folium.FeatureGroup("Reference route", show=True)
    folium.PolyLine(lines_of(trip, np.unique(trip.route), nodes_ll), color=GREEN, weight=6, opacity=0.65).add_to(ref)
    ref.add_to(m)

    pr = folium.FeatureGroup("Predicted route (classifier + HMM)", show=True)
    folium.PolyLine(latlon(pred["route_lonlat"]), color=BLUE, weight=3, opacity=0.95).add_to(pr)
    pr.add_to(m)

    gps = folium.FeatureGroup("GPS fixes", show=True)
    for k, (lon, lat) in enumerate(fixes):
        folium.CircleMarker([lat, lon], radius=2, color=ORANGE, fill=True, fill_opacity=0.9, weight=0,
                            tooltip=f"fix {k}").add_to(gps)
    gps.add_to(m)

    bad = folium.FeatureGroup("Fixes matched to the wrong road", show=True)
    for k in np.flatnonzero(wrong):
        lon, lat = fixes[k]
        folium.CircleMarker([lat, lon], radius=4, color=RED, fill=True, fill_opacity=1, weight=1,
                            tooltip=f"fix {k}: predicted arc {pred_arc[k]}, reference arc {ref_arc[k]}").add_to(bad)
    bad.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    top1 = 1 - wrong.sum() / n_lab
    legend = f"""
    <div style="position:fixed;bottom:28px;left:12px;z-index:9999;background:#fff;padding:10px 12px;border:1px solid #bbb;
                border-radius:6px;font:13px/1.5 Arial,sans-serif;color:#222">
      <b>PathSnap: trip {tid} ({partition})</b><br>
      <span style="color:{ORANGE}">&#9679;</span> GPS fixes ({len(fixes):,})<br>
      <span style="color:{RED}">&#9679;</span> fixes matched to the wrong road ({int(wrong.sum()):,})<br>
      <span style="color:{GREY}">&#9632;</span> road network<br>
      <span style="color:{GREEN}">&#9632;</span> reference route<br>
      <span style="color:{BLUE}">&#9632;</span> predicted route<br>
      Top-1 on {n_lab:,} labelled fixes: {top1 * 100:.1f}%
    </div>"""
    m.get_root().html.add_child(Element(legend))
    m.fit_bounds([[float(fixes[:, 1].min()), float(fixes[:, 0].min())], [float(fixes[:, 1].max()), float(fixes[:, 0].max())]])
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    m.save(out)
    print(f"saved {out} | fixes {len(fixes)} | roads {len(near)} | wrong {int(wrong.sum())} | top1 {top1:.4f}")


if __name__ == "__main__":
    build(*sys.argv[1:5])
