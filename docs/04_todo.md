# 04 · To-do list

Status as of 18 Nov 2026: docs and data audit done. **No model, HMM, API or UI code exists yet.** Put your name next to a task before starting. Tick it when merged.

Dependencies run top to bottom inside a stream; streams B and C can start once the table shapes in `03_plan.md` §13 are agreed, using a few trips.

## Stream A · Data, labels, candidates (blocks everything)
- [ ] A1. Loader and per-trip projection in `src/pathsnap/` (owner: )
- [ ] A2. Gap segmentation (`segment_id`), large-dt handling
- [ ] A3. EDA outputs for the report (missing-value table, trajectory plots, road-network plots, Folium map of a trip) on train and student-test, with the partition labelled
- [ ] A4. Label alignment (monotone, revisits allowed). Log ambiguity
- [ ] A5. Quarantine trips 33 and 54, document in code and report
- [ ] A6. Candidate generation (R, K, reverse twin). Candidate miss rate per trip

## Stream B · Features and models
- [ ] B1. Feature builder, scaler fitted on train only
- [ ] B2. Nearest-arc-with-heading baseline
- [ ] B3. Five models in different families, group-CV by trip, logged configs, seeds, times
- [ ] B4. Model comparison table and plots (CV macro F1, student-test Top-1 and macro F1, log loss)
- [ ] B5. Pick the final model with written reasoning
- [ ] B6. Optuna Bayesian optimisation on the chosen model, trial CSV, tuned vs untuned
- [ ] B7. Save final model, scaler, config in `models/`

## Stream C · HMM and Viterbi
- [ ] C1. Network graph per trip (sparse), Dijkstra with cutoff
- [ ] C2. Emissions, time-scaled transitions, gap reset
- [ ] C3. Log-space Viterbi, route output with connecting arcs
- [ ] C4. Tune decoder parameters on train and student-test only
- [ ] C5. Classifier-only vs classifier+HMM comparison, including failure cases
- [ ] C6. Metrics module: Top-1, macro F1, log loss, miss rate, route continuity, overlap

## Stream D · API, UI, maps, report
- [ ] D1. FastAPI app: `GET /trips`, `POST /predict`, request logging
- [ ] D2. Server-side scoring module that is the only reader of reference labels
- [ ] D3. UI page with trip list, map, results panel
- [ ] D4. Two-laptop test: bind address, CORS, firewall. Use student-test trips as stand-ins
- [ ] D5. Folium map generator, one HTML map committed
- [ ] D6. README with install, run and deploy steps
- [ ] D7. Report: tables and plots generated from `results/`; the nine questions per model and the analysis written by the team from its own results

## Everyone, before the event
- [ ] Fresh clone, fresh venv, run end to end, hit the API from the second laptop
- [ ] Viva rehearsal: each member explains dataset, preprocessing, features, model choice, BO, HMM, Viterbi, API, UI, results
- [ ] Scan history and files for secrets and data files before making the repo public
- [ ] Get answers to `05_open_questions.md`

## Risks
- Label alignment may be noisy or differ from the organisers' scoring.
- Candidate misses cap Top-1. Tune R and K first.
- Disconnected network pieces and time gaps break Viterbi. Handle with resets.
- Class imbalance: roughly one candidate in five to eight is correct. Macro F1 and log loss matter more than accuracy.
- Evaluator trips may have defects like trips 33 and 54. The pipeline must degrade gracefully.
- Cross-team similarity checks are real. Make your own feature and decoder choices and keep an honest commit history.
