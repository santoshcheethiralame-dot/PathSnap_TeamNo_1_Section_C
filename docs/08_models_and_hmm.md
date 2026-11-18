# 08 · Models and HMM results

All numbers come from the CSV files in `results/`. Labels are our own alignment (`docs/07_pipeline.md`), so the absolute values will shift if the organisers' labels differ. Student-test was used for reporting only, never for fitting, feature choice or decoder tuning.

## Setup
- Fit on the 78 non-quarantined train trips (33 and 54 excluded), at most 1,000 points per trip, about 457,000 candidate rows.
- 5-fold CV grouped by trip, with the three Moscow trips kept in one fold. Seed 0 everywhere.
- Candidate-level metrics treat each (point, candidate) as is-reference vs not. Top-1 is per point.
- Route continuity: share of consecutive distinct predicted arcs where one ends where the next starts. For reference, the aligned reference labels themselves score 93.2% on student-test (mean over trips, `results/reference_route_continuity_student_test.csv`), because sampling can skip an arc. So about 93% is the practical ceiling.

## Classifier comparison (`results/model_runs.csv`)
| Model | Family | CV Top-1 | CV macro F1 (std) | Student Top-1 | Student macro F1 | Student log loss | Route continuity | Fit time | Inference |
|---|---|---|---|---|---|---|---|---|---|
| nearest + heading rule | rule baseline | 87.3% (train mean) | n/a | 86.0% | n/a | n/a | n/a | n/a | n/a |
| HistGradientBoosting | boosting | 89.2% | 0.9329 (0.014) | **89.1%** | 0.931 | 0.094 | 74.5% | 2.4 s | 1.1 µs/row |
| Random Forest | bagged trees | 89.4% | 0.9343 (0.014) | 88.4% | 0.929 | 0.094 | 74.4% | 50 s | 2.9 µs/row |
| MLP (64, 32) | neural network | 89.0% | 0.9294 (0.014) | 88.6% | 0.929 | 0.097 | 74.9% | 112 s | 1.1 µs/row |

The three models are within about a point of each other and the fold-to-fold standard deviation of macro F1 is 0.014, so the ranking among them is not statistically solid. The gap to the baseline (86.0% to about 89%) is real. Logistic Regression and Gaussian NB (Koushik) will fill the remaining two rows.

**Provisional choice: HistGradientBoosting.** Best student-test Top-1, the same CV quality as the forest, and 20 to 45 times cheaper to train, which makes the Bayesian search affordable. Write your own justification in the report.

Epoch/iteration histories exist for HGB and the MLP only (`results/history_*.json`). The forest has none, so no training curve may be drawn for it. The HGB early-stopping hold-out and the MLP validation split use random rows, not whole trips, so they are for stopping only.

## HMM + Viterbi on top of HGB (`scripts/run_hmm.py`)
Settings tuned on train out-of-fold probabilities (the classifier never saw those trips), 54 combinations in `results/hmm_grid_hgb.csv`: emission weight 1.0, transition scale beta = 3 m, gamma = 0, flat penalty 20 for unreachable moves. The best setting sits inside the range searched (beta 2 to 20), and the surface is flat (train Top-1 92.0% to 92.1% for beta 2 to 5), so the choice is not sensitive.

| | Classifier only | Classifier + HMM |
|---|---|---|
| Train, out-of-fold Top-1 | 89.1% | 92.1% |
| Student-test Top-1 (point-weighted) | 89.1% | **92.8%** |
| Student-test route continuity (mean per trip) | 74.5% | **90.9%** |
| Student-test trips improved / worsened | n/a | 10 / 0 |

Per trip, Top-1 improves between 1.1 and 5.4 points (`results/hmm_student_test_hgb.csv`). The weakest trip is still 77 (84.3%), and trips 9, 52 and 91 have the lowest continuity, around 84% to 88%. HMM never made a student-test trip worse here, but the question "where did HMM not help" must be answered in the report from the train out-of-fold results too, and is not answered yet.

## How the decoder works (so you can explain it)
- **Hidden states:** the candidate arcs of each GPS fix.
- **Emission:** classifier probability, normalised over that fix's candidates, as a log.
- **Transition:** minus the gap between the driven route distance (shortest path on the directed road graph from the previous candidate's position to this one's) and the straight-line distance between the two fixes, divided by beta. A move with no path within the search limit gets a flat penalty, not probability zero.
- **Gaps:** the chain restarts wherever `segment_ids` changes (time gap over 60 s).
- **Cost:** route distances are computed once per trip and cached; the tuning grid then only re-runs Viterbi.
- Code: `src/pathsnap/hmm.py`, tests in `tests/test_hmm.py`.

## Caveats
- Everything depends on our label alignment.
- The model choice among the three is within noise.
- Decoder and classifier were both tuned and evaluated on trips from the same region mix; evaluator trips may behave differently.
- The Bayesian optimisation step (docs/04_todo.md B6) is run after this document was written; see `results/bayes_*` when it lands.
