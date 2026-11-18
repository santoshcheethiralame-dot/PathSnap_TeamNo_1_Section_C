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
- [ ] B3. Training harness: group-CV by trip, per-trip weighting, logged configs/seeds/times · **S**
- [ ] B3a. Random Forest, HistGradientBoosting/LightGBM, MLP · **S**
- [ ] B3b. Logistic Regression and Gaussian NB (after B3) · **K**
- [ ] B4. Comparison table and plots (CV macro F1, student-test Top-1 and macro F1, log loss) · **K**, table content from **S**
- [ ] B5. Final model choice with written reasoning · **S**
- [ ] B6. Optuna Bayesian optimisation, trial CSV, tuned vs untuned · **S**
- [ ] B7. Save final model, scaler and config in `models/` · **S**

## Stream C · HMM and Viterbi (Santosh)
- [ ] C1. Per-trip network graph, sparse Dijkstra with cutoff
- [ ] C2. Emissions, time-scaled transitions, gap reset
- [ ] C3. Log-space Viterbi, route output with connecting arcs
- [ ] C4. Tune decoder parameters on train and student-test only
- [ ] C5. Classifier-only vs classifier+HMM comparison, including failures
- [ ] C6. Metrics module: Top-1, macro F1, log loss, miss rate, route continuity, overlap
- [ ] C7. Put classifier+HMM into `predict_trip`

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
