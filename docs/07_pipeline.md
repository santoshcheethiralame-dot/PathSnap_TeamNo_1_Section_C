# 07 · Pipeline built so far

Code is in `src/pathsnap/`, tests in `tests/`, entry scripts in `scripts/`. Nothing here uses the evaluator partition: `trips.PARTITIONS` only lists `train` and `student_test`.

## Run it
```bash
pip install -r requirements.txt
python -m pytest -q tests
python scripts/build_dataset.py data/raw data/processed results/dataset_build_summary.csv
python scripts/run_baseline.py data/processed results/dataset_build_summary.csv results
```
The build takes about 20 seconds for all 90 trips and writes about 1.27 million candidate rows.

## Modules
| Module | What it does |
|---|---|
| `trips.py` | Loads the four files, projects to a per-trip local metric frame, builds arc geometry and reverse-twin lookup. `to_lonlat` converts back for maps. `with_route=False` never opens `.route`. |
| `segments.py` | Splits a trip into segments at gaps over 60 s or implied speeds over 80 m/s. |
| `labels.py` | Training labels. Viterbi over positions in the reference route: forward-only, revisits allowed (each route position is its own state), travelled route distance should match the GPS step distance. Fixes more than 100 m from the route stay unlabelled. Also decides quarantine. |
| `candidates.py` | For each point, the 10 nearest arcs within 40 m (retry at 80 m if none), plus the reverse twin of every kept arc. |
| `features.py` | One row per (point, candidate): distance, signed lateral offset, position along arc, arc length, heading agreement, speed, time gap, rank, candidate count, distance gap to best, node degrees. `FEATURE_COLUMNS` lists the model inputs. |
| `baseline.py` | Picks the candidate minimising `dist + w * (1 - cos_heading)`. |
| `predict.py` | Inference entry point for the API. Currently the baseline; the classifier and HMM replace the marked block. |

## Results (train + student-test; labels are our own alignment, not the organisers')
Source files: `results/dataset_build_summary.csv`, `results/baseline_weight_grid.csv`, `results/baseline_student_test.csv`.

**Labels.** 88 of 90 trips labelled cleanly: median label distance 4.7 m, median 100% of points labelled (minimum 90%). The quarantine rule flags exactly trips 33 and 54 (49% and 16% labelled). Three trips needed the alignment to restart where no forward move was possible (train 66 three times, train 81 once, student-test 56 once). Student-test trip 56, with 255 repeated arcs, labels 99.9% of points.

**Candidates.** On clean trips the reference arc is missing from the candidate set for 0.6% of labelled points (point-weighted), median trip 0.1%, worst trips 72 (14.8%), 61 (8.4%) and 43 (5.9%). The median trip has 6.7 candidates per point.

**Baseline** (heading weight 10, chosen on 78 non-quarantined train trips; Top-1 per point):
| | Train, 78 trips (mean per trip) | Student-test, 10 trips |
|---|---|---|
| nearest arc only | 57.4% | 52.4% (point-weighted) |
| nearest arc + heading | 87.3% | 86.0% (point-weighted), 87.2% (mean per trip) |

Per student-test trip, with heading: 72.5% (trip 77) to 94.2% (trip 0). Heading adds about 34 points. Anything we train must beat 86%, and the HMM must be judged against it too.

## Caveats
- Labels depend on our alignment and on the reference route's quality. The organisers' per-point labels may differ (`05_open_questions.md` #3).
- Candidate miss rate and the baseline were measured on the same proxy labels.
- The heading weight was picked on train only; student-test was used only to report.
- Trip 43 (labelled 97%, median distance 10.7 m) and trip 2 (labelled 90%) are the next-weakest trips after the two quarantined ones.
