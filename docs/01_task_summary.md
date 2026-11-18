# 01 · What the organisers require

Digest of the five organiser PDFs (Main Guidelines, Dataset Exploration, Models and Training Guidelines, Deliverables, Assessment and Plagiarism Policy). The PDFs are not in this repo. Re-read the originals before anything is final; this is a working summary, and where the PDFs contradict each other the PDFs win.

## The task
Given noisy GPS points from a trip and that trip's road network, reconstruct the road route the vehicle drove. Design: a classifier scores candidate road segments per GPS point, then an HMM with Viterbi decoding picks a connected route.

## Dataset
Per trip: four whitespace-separated files, no headers.

| File | Columns |
|---|---|
| `.track` | lon, lat, timestamp (s). Nominally 1 Hz |
| `.nodes` | lon, lat. Row index = node id |
| `.arcs` | from_node, to_node. Row index = `arc_id` |
| `.route` | one `arc_id` per line, in driving order (reference route) |

- A GPS row does not correspond to a row in any road file. Per-point labels must be derived.
- The reference route is "reference", not guaranteed truth. State its limits in the report.
- Partitions: **80 train**, **10 student-test**, **10 evaluator-test**. Never split a trip across partitions.
- We hold train and student-test only. Evaluator-test arrives on the day.

## Hard rules on data use
- Evaluator-test data and labels must not be used for training, feature selection, algorithm choice, hyperparameters, thresholds or decoder parameters.
- Fit every learned preprocessing step on training trips only.
- Tune decoder (HMM) parameters on train and student-test only.
- No hardcoded answers; no manual edits of predictions; ground truth stays server-side, never in the model or UI.
- Use a projected CRS or geodesic distance, never raw degrees.

## Modelling requirements
- Train and compare **at least 5 distinct models** across algorithm families, and document every model tried.
- Select on measured results and technical reasoning, never on training accuracy.
- **Bayesian** hyperparameter optimisation on the chosen model, with **group-aware CV** on the 80 training trips. Keep the trial log. Compare tuned vs untuned on the same metrics and partitions.
- HMM + Viterbi: explain hidden states, emissions, transitions, handling of gaps and missing candidates, and invalid (disconnected) transitions. Compare classifier-only vs classifier+HMM, including cases where HMM did not help.
- Metrics to report where applicable: accuracy, precision, recall, F1, confusion matrix, log loss, observation-level Top-1, candidate miss rate, route continuity, sequence-level performance, training time, inference time.
- Keep logs, seeds and configs.

## Final evaluation (live demo)
1. API (FastAPI) runs on **Laptop 1** with the models, HMM and road network.
2. UI runs on **Laptop 2** and lists the ten evaluator trips.
3. Evaluator picks a trip; the UI sends the trip ID to the API.
4. The API runs the full pipeline, scores itself against server-side reference data, returns route and results.
5. Must work for all ten trips. Show API logs on Laptop 1. No hardcoding, no bypassing the API.
6. Submitted source must match what is demonstrated.

## Deliverables
1. **Public GitHub repo** with: data processing, feature engineering, training code for every model, evaluation code/plots/confusion matrices, Bayesian optimisation code and results, HMM/Viterbi, API, UI, deployment config and README, Folium map code, supporting CSVs/logs. Include saved model weights and configs (size permitting). No passwords, keys or secrets.
2. **One consolidated PDF report**, sections:
   - 2.2 Dataset understanding and EDA (preprocessing, patterns, features and why; missing-value analysis, trajectory plots, road-network plots, at least one Folium map; say which partition each analysis used)
   - 3 Model experimentation: 3.1 comparison table (model, family, key config, CV macro F1, Top-1, route continuity, student-test macro F1, log loss, inference time), 3.2 the nine questions per model, 3.3 visualisations (CV macro F1 comparison, student-test Top-1/macro F1 comparison, log loss comparison, optional confusion matrices, train/val loss curves only for models that really have epochs, model-selection plots)
   - 4 Bayesian optimisation (six questions + tuned vs untuned table)
   - 5 HMM and Viterbi (six questions + classifier-only vs classifier+HMM)
   - 6 Folium maps (observations, road geometry, reference route, predicted route; legend, layer control; one screenshot in the PDF, HTML or generator code in the repo)
   - 7 API, UI, deployment (UI screenshots, successful API responses on the receiving laptop's CMD)
   - Every number must be reproducible from the repo.
3. A first-phase evaluation happens before lunch on the event day. The documents do not say what it checks.

## Academic integrity
- Reports and repos are checked for plagiarism, AI-generated text, and cross-team similarity (code, weights, configs, results, history).
- No sharing of models, weights, configs, code or results between teams.
- No fabricated metrics, graphs, citations or experiments. AI must not replace actual experimentation.
- Disclose AI assistance as the event requires.
- Every team member must be able to explain every part in a viva.

## Quirks in the PDFs
- Models and Training Guidelines numbering is broken (§9, then a second §5, §6, then §11). The Policy has two §5s. The report spec starts at §2.2.
- "Route continuity" and "macro F1" are not defined anywhere. See `05_open_questions.md`.
