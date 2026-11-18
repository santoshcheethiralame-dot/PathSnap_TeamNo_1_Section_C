# PathSnap · Team No. 1, Section C

Distributed map-matching and route reconstruction for UE24CS352A (Machine Learning), PES University. Noisy GPS points go in; a connected road route comes out. A classifier scores candidate road segments and an HMM with Viterbi decoding assembles the route. The API runs on one laptop and the evaluator UI on another.

## Status (18 Nov 2026)
Pipeline, five models, Bayesian optimisation, HMM/Viterbi, the saved model, the API and a first UI are in. After the context-feature upgrade (26 features; `PATHSNAP_FEATURES=v1` restores the original 17), student-test Top-1 goes from 86.0% (rule baseline) to 90.6% (tuned boosting) to 93.3% (boosting + HMM); route continuity 90.8%, route overlap 0.958 (our own definitions, see [docs/05_open_questions.md](docs/05_open_questions.md)). The Review 1 results (89.1% / 93.15%) are kept in `results/v1/`. Still open: report assembly, the two-laptop rehearsal on real hardware, and the open questions for the organisers. Details: [docs/04_todo.md](docs/04_todo.md), [docs/08_models_and_hmm.md](docs/08_models_and_hmm.md), [docs/09_error_analysis.md](docs/09_error_analysis.md).

## Start here (read in order)
1. [docs/01_task_summary.md](docs/01_task_summary.md): what the organisers require, rules and deliverables
2. [docs/02_data_audit.md](docs/02_data_audit.md): what the real data looks like, and what it changes
3. [docs/03_plan.md](docs/03_plan.md): proposed technical design and table shapes
4. [docs/06_work_split.md](docs/06_work_split.md): who owns what (Santosh, Koushik, Sharath)
5. [docs/07_pipeline.md](docs/07_pipeline.md): the data pipeline, how to run it, and its results
6. [docs/08_models_and_hmm.md](docs/08_models_and_hmm.md): model comparison and HMM results
7. [docs/04_todo.md](docs/04_todo.md): task checklist with owners
8. [docs/05_open_questions.md](docs/05_open_questions.md): questions for the organisers

## Set up
Python 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Put the organiser's dataset in `data/raw/` so it looks like this (this folder is gitignored; never commit data):

```
data/raw/train/<trip_id>/<trip_id>.{track,nodes,arcs,route}
data/raw/student_test/<trip_id>/<trip_id>.{track,nodes,arcs,route}
```

Evaluator-test data is not available during development and must never be used for training, tuning or selection.

Run the tests, the data audit, the table build and the baseline:

```bash
python -m pytest -q tests
python scripts/audit_data.py data/raw results/data_audit_per_trip.csv
python scripts/build_dataset.py data/raw data/processed results/dataset_build_summary.csv
python scripts/run_baseline.py data/processed results/dataset_build_summary.csv results
```

## Run the API and UI (two laptops)
The API loads the saved model in `models/` and serves whatever trip folders are in one directory. It never opens the reference route during prediction; the scoring module (`api/scoring.py`) is the only reader of `.route` and its numbers are returned with the prediction.

**Laptop 1 (API).** Set the trips folder (the folder whose sub-folders are the trips) and start the server on all interfaces:

```bash
set PATHSNAP_TRIPS_DIR=D:\evaluator_trips                    # Windows cmd, example path
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

If `PATHSNAP_TRIPS_DIR` is not set it serves `../Student folder/student_test`. Trips are listed from the folder at request time, so no code change is needed on the day. A trip with no `.route` file is still predicted; it just comes back with `"scored": false`. Open TCP port 8000 in the Windows firewall (Private network) and find the laptop's address with `ipconfig`. Every request and every prediction (trip id, method, route size, Top-1, continuity, time) is printed to the console and written to `api/logs/requests.log`.

**Laptop 2 (UI).** Copy the `ui/` folder, or serve it:

```bash
python -m http.server 8080 --directory ui
```

Open `http://localhost:8080`, type Laptop 1's address (for example `http://192.168.1.20:8000`), press Load Trips, pick a trip, press Reconstruct Route. The map and the four result cards come straight from the API response. The page loads Leaflet and the map tiles from the internet.

Endpoints: `GET /health`, `GET /trips`, `POST /predict` with `{"trip_id": "..."}`. Interactive docs at `/docs`. Tests (no organiser data needed): `python -m pytest -q tests`.

## Reproduce the results
Commands are in each script's docstring; `docs/07_pipeline.md` and `docs/08_models_and_hmm.md` give the order and the results.

```bash
python scripts/audit_data.py data/raw results/data_audit_per_trip.csv
python scripts/build_dataset.py data/raw data/processed results/dataset_build_summary.csv
python scripts/train_models.py DATA_RAW DATA_PROCESSED SUMMARY_CSV OUT_DIR PROBS_DIR rf hgb mlp lr gnb
python scripts/bayes_opt.py DATA_PROCESSED SUMMARY_CSV OUT_DIR
python scripts/run_hmm.py DATA_RAW DATA_PROCESSED SUMMARY_CSV PROBS_DIR CACHE_DIR OUT_DIR hgb_tuned
python scripts/export_model.py DATA_PROCESSED SUMMARY_CSV HMM_SUMMARY_CSV hgb_tuned
python scripts/check_predict.py data/raw                      # end to end, reproduces 93.3%
python scripts/make_folium_map.py data/raw student_test 00000009 report/maps/trip_00000009_student_test.html
```

## Repository contents
| Path | What |
|---|---|
| `docs/` | Task summary, data audit, plan, to-do, open questions, work split, pipeline, models and HMM, error analysis, report draft |
| `src/pathsnap/` | Loader, segments, labels, candidates, features, models, HMM, metrics, `predict_trip` |
| `api/` | FastAPI app and the server-side scoring module |
| `ui/` | Single-page UI (Leaflet map, trip list, result cards) |
| `models/` | Saved tuned boosting model and decoder settings (`model.joblib`, `config.json`) |
| `tests/` | Unit tests on synthetic roads and a synthetic trip for the API |
| `scripts/` | Audit, dataset build, training, Bayesian search, HMM, analysis, figures, maps |
| `results/` | Every CSV and JSON behind the reported numbers |
| `report/` | `PathSnap_Report.pdf` (final report, built by `scripts/make_report.py`), figures (EDA, models, UI screenshots, Review 1), Folium map, Review 1 handout |
| `requirements.txt` | Pinned dependencies |

## Ground rules
- Never commit the dataset, passwords, keys or other secrets.
- Do not use evaluator-test data for anything in development.
- Fit learned preprocessing on training trips only; tune on train and student-test only.
- Every reported number must be reproducible from this repo. No made-up metrics, curves or maps.
- Every member must be able to explain all code, experiments and results in the viva.
- Work in short-lived branches and small commits so the history reflects real work.

## AI-assistance disclosure
The documents in `docs/`, `README.md`, `scripts/` and `src/pathsnap/` were drafted with the help of Claude (Anthropic) from the organiser PDFs and the team's dataset, and every reported number was produced by running the code in this repo. The docs are working notes for the team, not report text. The experiments, analysis and final report must be the team's own work.
