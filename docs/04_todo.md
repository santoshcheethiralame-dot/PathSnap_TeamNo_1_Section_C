# 04 · To-do list

Status as of 18 Nov 2026. Owners are set in [06_work_split.md](06_work_split.md): **S** = Santosh, **K** = Koushik, **Sh** = Sharath. Tick a task when it is merged to `main`. Streams B and C wait on the training harness (B3), nothing else blocks anyone.

## Stream A · Data, labels, candidates (Santosh) · DONE
- [x] A1. Loader and per-trip projection (`trips.py`)
- [x] A2. Gap segmentation (`segments.py`)
- [ ] A3. EDA outputs for the report, with the partition labelled · **K**
- [x] A4. Label alignment, monotone with revisits (`labels.py`)
- [x] A5. Quarantine rule, trips 33 and 54 flagged automatically
- [x] A6. Candidate generation, miss rate per trip (`candidates.py`, `results/dataset_build_summary.csv`)

## Stream B · Features and models
- [x] B1. Feature builder (`features.py`) · S. Scaler fitting still to do inside the harness
- [x] B2. Nearest-arc-with-heading baseline: 86.0% student-test Top-1 (`07_pipeline.md`) · S
- [x] B3. Training harness (`scripts/train_models.py`, `src/pathsnap/modeling.py`): group-CV by trip, 1,000-point cap per trip, logged configs/seeds/times · S
- [x] B3a. Random Forest, HistGradientBoosting, MLP: student-test Top-1 88.4% / 89.1% / 88.6% (`results/model_runs.csv`) · S
- [x] B3b. Logistic Regression and Gaussian NB trained (added by S so the five-model set exists for Review 1; results in `results/model_runs.csv`). **K** still owns the report write-up for these two: the nine questions in report §3.2
- [x] B4. Comparison plots drafted by `scripts/make_model_plots.py` into `report/figures/models/`. **K** reviews and restyles them for the report
- [x] B5. Final model choice with written reasoning: tuned HGB + HMM (`08_models_and_hmm.md`). The student must still write their own justification in the report · S
- [x] B6. Optuna Bayesian optimisation: 40 trials, CV Top-1 89.2% to 89.5%, student-test unchanged (`results/bayes_*`) · S
- [x] B7. Final model and config saved in `models/` (model.joblib 1.4 MB, config.json; no scaler needed for trees) · S

## Stream C · HMM and Viterbi (Santosh)
- [x] C1. Per-trip network graph, sparse Dijkstra with cutoff (`hmm.py`)
- [x] C2. Emissions, route-distance transitions, gap reset
- [x] C3. Log-space Viterbi, plus connecting-arc route fill for the map polyline (`route_nodes`)
- [x] C4. Decoder parameters tuned on train out-of-fold probabilities (`results/hmm_grid_*.csv`)
- [x] C5. Classifier-only vs classifier+HMM: student-test 89.1% to 93.2% Top-1, 75.2% to 90.4% continuity. Failure cases: HMM worse on 1 of 78 train trips (trip 72), see `08_models_and_hmm.md`. The cause of trip 16's low accuracy is not analysed
- [x] C6. Metrics module (`metrics.py`). Route overlap metric not done
- [x] C7. Classifier + HMM wired into `predict_trip`; end-to-end check reproduces 93.15% (`scripts/check_predict.py`). Sharath can now build the API on it

## Stream D · API, UI, maps, report
- [ ] D1. FastAPI app: `GET /trips`, `POST /predict`, request logging · **Sh**
- [ ] D2. Server-side scoring module, the only reader of `.route` · **Sh**
- [ ] D3. UI page with trip list, map, results panel · **Sh**
- [ ] D4. Two-laptop test: bind address, CORS, firewall · **Sh**
- [ ] D5. Folium map generator, one HTML map committed · **K**
- [ ] D6. README install, run and deploy steps · **Sh**
- [ ] D7. Report: tables and plots generated from `results/`, analysis written by the owner of each section (see split doc)

## Everyone, before the event
- [ ] Fresh clone, fresh venv, run end to end, hit the API from the second laptop · Sh with all
- [ ] Walkthroughs and viva rehearsal (see split doc)
- [ ] Scan history and files for secrets and data files before making the repo public · Sh
- [ ] Get answers to `05_open_questions.md` · S

## Risks
- Label alignment may differ from the organisers' scoring.
- Candidate misses cap Top-1 (0.6% pooled on clean trips, up to 14.8% on trip 72).
- Disconnected network pieces and time gaps break Viterbi. Handle with resets.
- Class imbalance: roughly one candidate in five to eight is correct. Macro F1 and log loss matter more than accuracy.
- Evaluator trips may have defects like trips 33 and 54. The pipeline must degrade gracefully.
- Cross-team similarity checks are real. Make your own feature and decoder choices and keep an honest commit history.
