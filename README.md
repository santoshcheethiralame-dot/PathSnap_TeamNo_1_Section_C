# PathSnap · Team No. 1, Section C

Distributed map-matching and route reconstruction for UE24CS352A (Machine Learning), PES University. Noisy GPS points go in; a connected road route comes out. A classifier scores candidate road segments and an HMM with Viterbi decoding assembles the route. The API runs on one laptop and the evaluator UI on another.

## Status (18 Nov 2026)
Data audit and the front half of the pipeline are done: loader, label alignment, candidates, features and a nearest-arc-with-heading baseline (86.0% Top-1 on student-test). **No trained model, HMM, API or UI yet.** Who does what next is in [docs/06_work_split.md](docs/06_work_split.md).

## Start here (read in order)
1. [docs/01_task_summary.md](docs/01_task_summary.md): what the organisers require, rules and deliverables
2. [docs/02_data_audit.md](docs/02_data_audit.md): what the real data looks like, and what it changes
3. [docs/03_plan.md](docs/03_plan.md): proposed technical design and table shapes
4. [docs/06_work_split.md](docs/06_work_split.md): who owns what (Santosh, Koushik, Sharath)
5. [docs/07_pipeline.md](docs/07_pipeline.md): the code built so far, how to run it, and its results
6. [docs/04_todo.md](docs/04_todo.md): task checklist with owners
7. [docs/05_open_questions.md](docs/05_open_questions.md): questions for the organisers

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

## Repository contents
| Path | What |
|---|---|
| `docs/` | Task summary, data audit, plan, to-do, open questions, work split, pipeline notes |
| `src/pathsnap/` | Loader, segments, labels, candidates, features, baseline, `predict_trip` |
| `tests/` | Unit tests on synthetic roads |
| `scripts/` | Audit, dataset build, baseline |
| `results/` | Audit, build summary and baseline CSVs |
| `requirements.txt` | Pinned dependencies |

More folders (`api/`, `ui/`, `models/`, `report/`) arrive as work lands. See the planned layout in [docs/03_plan.md](docs/03_plan.md) §12.

## Ground rules
- Never commit the dataset, passwords, keys or other secrets.
- Do not use evaluator-test data for anything in development.
- Fit learned preprocessing on training trips only; tune on train and student-test only.
- Every reported number must be reproducible from this repo. No made-up metrics, curves or maps.
- Every member must be able to explain all code, experiments and results in the viva.
- Work in short-lived branches and small commits so the history reflects real work.

## AI-assistance disclosure
The documents in `docs/`, `README.md`, `scripts/` and `src/pathsnap/` were drafted with the help of Claude (Anthropic) from the organiser PDFs and the team's dataset, and every reported number was produced by running the code in this repo. The docs are working notes for the team, not report text. The experiments, analysis and final report must be the team's own work.
