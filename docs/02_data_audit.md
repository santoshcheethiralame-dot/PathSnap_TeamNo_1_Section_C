# 02 · Data audit (what the real data looks like)

Audited on **train (80 trips) and student-test (10 trips)** only, 227,577 GPS points in total (208,417 train, 19,160 student-test). Evaluator-test was not available and was not touched.

Reproduce: `python scripts/audit_data.py data/raw results/data_audit_per_trip.csv`. Every number below comes from `results/data_audit_per_trip.csv` (one row per trip, 55 columns).

**Caveat.** The "label" used here is a proxy: the nearest arc *of the reference route* to each point. It is only good enough to size the candidate search. The real label alignment is task 4 in `04_todo.md`.

## 1. Size and structure
| | min | median | 95th pct | max |
|---|---|---|---|---|
| GPS points per trip | 262 | 1,497 | 7,516 | 19,313 |
| Network nodes per trip | 299 | 9,305 | 94,145 | 155,003 |
| Network arcs per trip | 649 | 18,762 | 184,034 | 307,392 |
| Arcs in reference route | 59 | 273 | 950 | 2,266 |
| Median arc length (m) | 6.8 | 25.2 | 57.8 | 85.1 |

- All files present for all 90 trips. No NaN values, no invalid node ids in arcs, no invalid `arc_id` in routes, no isolated nodes.
- **Every trip has its own network** (different node and arc counts, different ids). `arc_id` means nothing across trips. Never use it as a class label or a feature.
- Arcs are straight node-to-node segments, directed. Median **95%** of arcs have an exact reverse twin (min 35%). Both directions of a road sit at the same distance from a GPS point.
- Only one place is shared by several trips (Moscow: train trips 10, 16, 29). Student-test shares no area with train.

## 2. Coordinates
Trips are **worldwide** (Russia, USA, Spain, Japan, Brazil, Argentina, South Africa, Philippines, ...). One fixed UTM zone does not work. Project each trip separately around its own centroid (the script uses a local transverse Mercator in metres). Order in the files is lon, lat. All track points fall inside their trip's network bounding box.

## 3. Timing: not really 1 Hz
- Median trip: 99.1% of steps are exactly 1 s. But **14 of 90 trips have fewer than 90%**: train 4, 14, 18, 22, 25, 26, 39, 46, 71, 78, 81, 90, 98 and student-test 76. Trips 4, 22, 39, 46 are about 53-62% regular and some steps are 10-17 s.
- Huge gaps: train trip 90 has a **153,064 s** step, trip 78 has 52,810 s, trips 21 and 69 about 2,200-2,700 s.
- Some timestamps are fractional (7.2, 2.003). The first timestamp is not always 1. No timestamp ever goes backwards or repeats.
- Largest jump between consecutive points ranges up to 888 m, and the 95th-percentile trip has 111 steps over 50 m.

Consequence: speed, windowed features and HMM transitions must use the real time delta. A very large gap must reset the HMM chain.

## 4. The reference route
- Median distance from a GPS point to the route is **4.5 m** (95th pct 14 m). Student-test trips are all in the 4.2-6.8 m range.
- 11 trips have 1-3 **breaks in the route chain** (an arc whose end node is not the next arc's start node). Route continuity of the reference is therefore not always 100%.
- 10 trips **repeat arcs** (loops or revisits). Student-test trip 56 repeats 255 arcs, train 69 repeats 29, train 92 repeats 14.
- Nearest-route-arc labels move backwards along the route a median of 23 times per trip (up to 738), so a plain nearest-arc label is not reliable. Use a monotone alignment that allows revisits.
- Median 89% of moving points agree in direction with the route arc they are nearest to (min 51%). Direction is a real signal, but noisy at low speed.

## 5. Two defective training trips
| Trip | Points | Share of train points | What is wrong |
|---|---|---|---|
| train 33 | 5,844 | 2.8% | Reference route skips about half of the drive. 84% of points sit within 15 m of *some* road, but only 24% within 15 m of the *route*. Points in deciles 5-9 are about 135 m from it. |
| train 54 | 19,313 | 9.3% | The road network is missing most of the drive. Only 13% of points are within 15 m of *any* arc. Median distance to route 139 m. |

Together they are **12% of all training points**. They would inject label noise. Recommendation: quarantine both from training and CV, say so in the report with this table, and keep them in a separate "known-defective" analysis. Evaluator trips may contain similar defects; the pipeline must not crash on them.

Worth watching but not defective: trips 4 (68% of points within 15 m of route), 43 (64%) and 80 (77%) have noisy or sparse GPS but the route itself is fine.

## 6. Candidate search (proxy label)
Clean trips only (all except 33 and 54):

| Radius R | Reference arc inside R (median trip) | Same, point-weighted | Mean candidates per point | Points with no arc inside R |
|---|---|---|---|---|
| 15 m | 95.9% | 92.4% | 3.1 | 5.9% |
| 25 m | 99.6% | 97.2% | 4.8 | 2.3% |
| 40 m | 100% | 98.6% | 7.6 | 1.1% |
| 60 m | 100% | 99.0% | 12.1 | 0.6% |
| 100 m | 100% | 99.6% | 23.7 | 0.15% |

Rank of the reference arc among arcs ordered by distance: top-1 56% (median trip), top-3 96.5%, top-5 99.1%, top-10 99.96%.

**Nearest-arc Top-1 is only about 56%** because the nearest arc is usually one of two directions of the same road, a coin flip without heading. On student-test it ranges 0.40-0.81 per trip. So the baseline and the classifier need heading and direction features.

Starting point: R about 40 m, K about 10, force-include the reverse twin of every candidate. Choose finally from the miss rate on training trips.

## 7. Consequences for the build
1. Per-trip local projection, not a fixed CRS.
2. Use the real time delta everywhere. Reset the HMM at large gaps.
3. Direction features are mandatory.
4. Label alignment must be monotone and allow revisits.
5. Quarantine trips 33 and 54.
6. Trip sizes vary 74x (262 to 19,313 points). Weight or cap points per trip so one trip does not dominate training, and report metrics per trip as well as pooled.
7. Group cross-validation by trip, with Moscow trips 10, 16, 29 in the same fold.
8. Networks reach 307k arcs. Use a spatial index for candidates and a sparse shortest-path with a distance cutoff for HMM transitions. Loading and indexing one trip must be fast enough for the live demo.
