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
| Logistic Regression | linear | 88.4% | 0.9149 (0.015) | 86.5% | 0.909 | 0.128 | 72.4% | 1.5 s | 0.17 µs/row |
| Gaussian Naive Bayes | probabilistic | 86.2% | 0.8390 (0.013) | 83.3% | 0.827 | 0.687 | 71.6% | 0.2 s | 0.41 µs/row |
| HistGradientBoosting | boosting | 89.2% | 0.9329 (0.014) | **89.1%** | 0.931 | 0.094 | 74.5% | 2.4 s | 1.1 µs/row |
| Random Forest | bagged trees | 89.4% | 0.9343 (0.014) | 88.4% | 0.929 | 0.094 | 74.4% | 50 s | 2.9 µs/row |
| MLP (64, 32) | neural network | 89.0% | 0.9294 (0.014) | 88.6% | 0.929 | 0.097 | 74.9% | 112 s | 1.1 µs/row |

The forest, boosting and MLP are within about a point of each other, and the fold-to-fold standard deviation of macro F1 is 0.014, so the ranking among them is not statistically solid (paired over the same folds: boosting minus forest is -0.18 points Top-1, boosting minus MLP +0.17). Boosting beats logistic regression in all 5 folds, by 0.8 points on average. The gap to the baseline (86.0% to about 89%) is real; logistic regression is only 0.5 points above the baseline and naive Bayes is below it.

## Overfitting and error evidence (`results/model_diagnostics.csv`, `confusion_matrices.csv`)
| Model | Train Top-1 | Student Top-1 | Train log loss | Student log loss | Student precision / recall |
|---|---|---|---|---|---|
| Logistic Regression | 87.9% | 86.5% | 0.125 | 0.128 | 0.841 / 0.853 |
| Gaussian NB | 85.9% | 83.3% | 0.580 | 0.687 | 0.608 / 0.888 |
| Random Forest | **95.1%** | 88.4% | 0.047 | 0.094 | 0.896 / 0.865 |
| HistGradientBoosting | 90.6% | 89.1% | 0.077 | 0.094 | 0.886 / 0.883 |
| MLP | 90.1% | 88.6% | 0.080 | 0.097 | 0.885 / 0.877 |

The forest overfits (6.7-point train-to-test gap). Logistic regression underfits (low on both). Naive Bayes is badly calibrated: many false positives, log loss 0.69, because the features are strongly correlated. On student-test, boosting's correct road is missing from the candidates for 49 of 19,157 labelled points (0.26%); of its remaining mistakes (10.6% of points), 32% pick the opposite direction of the correct road and 68% a different segment. What that 68% consists of has not been analysed.

Figures: `report/figures/models/` (made by `scripts/make_model_plots.py`). Review scripts: `docs/presentation/`.

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
