# 06 · Work split

Three people. Santosh carries the core pipeline (the critical path). Koushik and Sharath each own a smaller, self-contained piece that can start today. Rough effort: Santosh about 65%, Sharath about 20%, Koushik about 15%.

Everyone must still be able to explain **every** part in the viva. See "Knowledge sharing" at the bottom.

## Santosh · core pipeline
Already done and pushed (see `07_pipeline.md`): loader, per-trip projection, segments, label alignment, candidates, features, nearest-arc-with-heading baseline, `predict_trip` entry point, tests.

Still to do, in this order:
1. **Training harness** (`scripts/train_models.py`): group-CV by trip, per-trip weighting, quarantined trips excluded, logs configs, seeds, times, metrics to `results/`. *Koushik is blocked on this for his models.*
2. Three models: Random Forest, HistGradientBoosting/LightGBM, MLP.
3. Model comparison and selection, with written reasoning.
4. Optuna Bayesian optimisation on the chosen model.
5. HMM and Viterbi, decoder tuning, classifier-only vs classifier+HMM.
6. Metrics module (Top-1, macro F1, log loss, miss rate, route continuity, overlap).
7. Swap the classifier+HMM into `predict_trip` (the marked block). The API keeps working unchanged.
8. Report: §3.2 for his models, §4, §5.

## Koushik · EDA, maps, two simple models
Start now (1 and 2), then 3 once the harness lands.
1. **EDA figures** for the report, from `src/pathsnap/trips.py` and `results/data_audit_per_trip.csv`. Script `scripts/make_eda.py`, output to `report/figures/eda/`. Every figure title states the partition (train or student-test).
   - missing-value table (there are none; show that you checked)
   - GPS trajectory plots for several trips
   - road-network plots for several trips
   - histogram of time gaps; histogram of GPS-to-route distance
   - a short written data-quality log (use `02_data_audit.md` as the starting point, verify it yourself)
2. **Folium map generator**, `scripts/make_map.py TRIP_ID`. Layers: raw GPS points, reference route, nearby road network, predicted route (from `pathsnap.predict.predict_trip`). Distinct colours, legend, layer control, sensible zoom. Commit one HTML map and a PNG screenshot. Use `Trip.to_lonlat` for coordinates.
3. **Two models**, Logistic Regression and Gaussian Naive Bayes, using the harness once it exists: features from `pathsnap.features.FEATURE_COLUMNS`, group-CV by trip.
4. **Comparison plots** (CV macro F1, student-test Top-1 and macro F1, log loss) from the results CSVs.
5. Report: §2.2 (EDA), §6 (Folium), §3.2 answers for LR and NB.

## Sharath · API, UI, deployment
Start now. `pathsnap.predict.predict_trip(trip_dir)` already works (baseline predictor), so you can build against real output from day one.
1. **FastAPI app** `api/main.py`: `GET /health`, `GET /trips` (trip ids from a configured folder; use student-test for development), `POST /predict {trip_id}`. Log every request to console and file. CORS on. Bind `0.0.0.0`.
2. **Scoring module** `api/scoring.py`: the only code that ever opens a `.route` file. It aligns labels with `pathsnap.labels.align_labels`, computes Top-1 and the other agreed metrics, and returns numbers only. Reference labels must never reach the predictor, the UI or the API response.
3. **UI** `ui/index.html`: trip list, calls the API, Leaflet map with fixes and predicted route, accuracy panel, configurable API address.
4. **Two-laptop test**: IP, firewall port, CORS. Document the steps in the README. Screenshots of the UI and of successful responses in the API laptop's CMD, for report §7.
5. **Release hygiene**: scan the repo and its history for secrets and data files before the repo goes public; fresh-clone dry run before the event.
6. Report: §7.

## Interfaces you can rely on
- `pathsnap.trips.load_trip(dir, with_route=False)` for inference, `with_route=True` only for training or scoring.
- `pathsnap.predict.predict_trip(dir)` returns `point_arc`, `route_arcs`, `route_lonlat`, `fix_lonlat`.
- Feature tables are in `data/processed/<partition>/<trip>.parquet` after `python scripts/build_dataset.py ...` (takes about 20 seconds). Columns are listed in `pathsnap.features`.
- Change these shapes only by agreement.

## How we work
- One short-lived branch per task, small commits, pull request into `main`, Santosh reviews.
- Write your own code and your own report text. Do not copy from other teams or from generated text you do not understand. AI help must be disclosed as per the event rules.
- Never commit the dataset or secrets.

## Knowledge sharing (viva)
Before the event each person gives a 20-minute walkthrough of their part to the other two: Santosh (labels, candidates, features, models, HMM), Koushik (EDA, maps, his two models), Sharath (API, UI, deployment). Then each person practises answering questions on the parts they did not write.

## Open points that affect you
- Metric definitions (macro F1, route continuity) are still unconfirmed: `05_open_questions.md` #4 and #5. Sharath should keep the scoring module easy to change.
