# PathSnap · Team No. 1, Section C

Distributed map-matching and route reconstruction for UE24CS352A (Machine Learning), PES University. Noisy GPS points go in; a connected road route comes out. A classifier scores candidate road segments and an HMM with Viterbi decoding assembles the route. The API runs on one laptop and the evaluator UI on another.

## Status (18 Nov 2026)
Planning and data audit are done. **No model, HMM, API or UI code exists yet.** The next step is the work in [docs/04_todo.md](docs/04_todo.md).

## Start here (read in order)
1. [docs/01_task_summary.md](docs/01_task_summary.md): what the organisers require, rules and deliverables
2. [docs/02_data_audit.md](docs/02_data_audit.md): what the real data looks like, and what it changes
3. [docs/03_plan.md](docs/03_plan.md): proposed technical design and table shapes
4. [docs/04_todo.md](docs/04_todo.md): task list split into four streams. Put your name on a task before starting
5. [docs/05_open_questions.md](docs/05_open_questions.md): questions for the organisers

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

Re-run the data audit:

```bash
python scripts/audit_data.py data/raw results/data_audit_per_trip.csv
```

## Repository contents
| Path | What |
|---|---|
| `docs/` | Task summary, data audit, plan, to-do, open questions |
| `scripts/audit_data.py` | Per-trip audit of the raw dataset |
| `results/data_audit_per_trip.csv` | Output of the audit, one row per trip |
| `requirements.txt` | Pinned dependencies used so far |

More folders (`src/`, `api/`, `ui/`, `models/`, `report/`) arrive as work lands. See the planned layout in [docs/03_plan.md](docs/03_plan.md) §12.

## Ground rules
- Never commit the dataset, passwords, keys or other secrets.
- Do not use evaluator-test data for anything in development.
- Fit learned preprocessing on training trips only; tune on train and student-test only.
- Every reported number must be reproducible from this repo. No made-up metrics, curves or maps.
- Every member must be able to explain all code, experiments and results in the viva.
- Work in short-lived branches and small commits so the history reflects real work.

## AI-assistance disclosure
The documents in `docs/`, `README.md` and `scripts/audit_data.py` were drafted with the help of Claude (Anthropic) from the organiser PDFs and the team's dataset, and the numbers were produced by running the audit script. They are working notes for the team. They are not report text, and the experiments, analysis and final report must be the team's own work.
