# 03 · Technical plan

This is a proposal, not a decision. Nothing here has been implemented or measured yet. The team owns the final choices and must be able to defend them in the viva.

## 1. Framing
- A GPS point is matched to a **candidate arc** among nearby arcs. A classifier scores each (point, candidate) pair as "is the reference arc".
- Observation-level Top-1: the highest-scoring candidate per point.
- HMM hidden states: the candidate arcs of each point. Emissions come from the classifier's probabilities. Transitions come from the road network. Viterbi picks the most coherent route.
- A nearest-arc-with-heading rule is the baseline everything is compared with (not one of the five models).

## 2. Data handling
1. Load a trip: four files, per-trip local metric projection (see `02_data_audit.md` §2).
2. Gaps: any step with a large time delta ends a segment. Features and HMM treat segments independently.
3. Never read the evaluator partition during development. Training code takes explicit train and student-test paths only.

## 3. Candidates
- Spatial index over arc segments. For each point take arcs within R (start 40 m), cap at K nearest (start 10), always add the reverse twin.
- Points with no candidate: widen the radius once, else mark as a gap.
- Report candidate miss rate on train and student-test.

## 4. Labels (training only)
- Align each GPS point to a route arc with dynamic programming over the route order: monotone non-decreasing position, revisits allowed, cost = distance to the arc (plus a heading term).
- Log how many points are ambiguous and how many are further than 50 m from the aligned arc.
- Skip quarantined trips 33 and 54.

## 5. Features (proposal)
Per (point, candidate): distance to arc, position along the arc, heading difference between motion and arc direction, speed (from real dt), arc length, node degrees, rank by distance, distance gap to the second-best candidate, smoothed heading over a short window, time gap to neighbours, whether the candidate is the reverse twin of the nearest one. Fit any scaler on training trips only.

## 6. Models (five or more, different families)
| Family | Candidate |
|---|---|
| Linear | Logistic Regression |
| Probabilistic | Gaussian Naive Bayes |
| Bagging | Random Forest |
| Boosting | HistGradientBoosting or LightGBM |
| Neural | MLP |

kNN and RBF-SVM are too slow for about 1M candidate rows without subsampling. Every model: group-aware CV by trip, config, seed, training time and inference time logged. Pick the final model on CV and student-test results, never training accuracy.

## 7. Bayesian optimisation
Optuna (TPE) on the chosen model. GroupKFold by trip on the training trips, objective stated and justified, about 40 trials, trial table saved as CSV, refit, compare tuned vs untuned on student-test with identical metrics and partitions.

## 8. HMM and Viterbi
- **Emission:** classifier probability per candidate, normalised across the point's candidates.
- **Transition:** Newson-Krumm style: penalise the difference between shortest-path network distance of consecutive candidates and the straight-line distance between the two GPS points. Disconnected candidate pairs get a large penalty, not a hard zero.
- **Scale by time:** allowed travel distance grows with dt.
- **Gaps and missing candidates:** reset the chain at large gaps and restart.
- **Decoding:** log-space Viterbi. Output a predicted arc per point and a route with connecting shortest-path arcs.
- Network distances: sparse Dijkstra with a distance cutoff.
- Tune decoder parameters on train and student-test only. Compare classifier-only vs classifier+HMM, and write up where HMM hurt.

## 9. Metrics
Observation-level Top-1, candidate-level macro F1, log loss, candidate miss rate, route continuity, sequence-level overlap with the reference, training and inference time. Definitions proposed here are pending `05_open_questions.md` #4 and #5. Report per trip as well as pooled.

## 10. API and UI
- FastAPI on Laptop 1. `GET /trips` returns trip IDs. `POST /predict {trip_id}` runs the full pipeline and returns the route, per-point predictions and accuracy.
- Reference labels are read only by the scoring module, server-side, never by the model or UI. Log every request (the evaluator will ask for logs).
- UI on Laptop 2: static page, trip list, Leaflet map of observed vs predicted route, accuracy panel. Bind the API to `0.0.0.0`, enable CORS, open the firewall port.
- Test across two real machines using student-test trips as stand-ins for evaluator trips, through the same code path as the final system.

## 11. Folium and report plumbing
- One script produces maps (observations, road geometry, reference, prediction, legend, layer control).
- Report tables and plots are generated from results CSVs, so every number is traceable to the repo.

## 12. Proposed repo layout
```
data/                gitignored. raw/train, raw/student_test
src/pathsnap/        io, projection, labels, candidates, features, models, bayes, hmm, metrics
scripts/             audit, train, evaluate, make_maps
api/                 FastAPI app, scoring module
ui/                  evaluator page
results/             CSVs, logs, trial tables (committed)
models/              final weights and config (committed if small)
docs/                these notes
report/              figures and report sources
```

## 13. Interface contract between workstreams
Work can split if these artefacts keep a fixed shape. Change them only by agreement.

**`candidates` table** (one row per point-candidate pair): `trip_id, point_idx, arc_id, rank, dist_m, offset_frac, is_twin` plus feature columns, plus `label` (0/1) for train and student-test only. Store as Parquet (add `pyarrow`) or gzipped CSV.

**`points` table**: `trip_id, point_idx, t, x_m, y_m, segment_id` (segment_id changes at gaps).

**Predictions**: `trip_id, point_idx, arc_id, score` from the classifier; `trip_id, point_idx, arc_id` from the HMM.
