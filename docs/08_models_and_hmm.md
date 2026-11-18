# 08 · Models and HMM results

All numbers come from the CSV files in `results/`. Labels are our own alignment (`docs/07_pipeline.md`), so the absolute values will shift if the organisers' labels differ. Student-test was used for reporting and for the development-stage model comparison the rules allow, never for fitting, feature choice, Bayesian search or decoder tuning.

## Setup
- Fit on the 78 non-quarantined train trips (33 and 54 excluded), at most 1,000 points per trip, about 457,000 candidate rows.
- 5-fold CV grouped by trip, with the three Moscow trips kept in one fold. Seed 0 everywhere.
- Candidate-level metrics treat each (point, candidate) as is-reference vs not. Top-1 is per point.
- Route continuity: share of consecutive distinct predicted arcs where one ends where the next starts. For reference, the aligned reference labels themselves score 93.2% on student-test (mean over trips, `results/reference_route_continuity_student_test.csv`), because sampling can skip an arc. So about 93% is the practical ceiling.

## FINAL results (26 features: 17 original + 9 context, `results/`)
The Review 1 run (17 features) is kept in `results/v1/` and described further down. The context features (heading carried over stops, signed movement along each candidate arc over ±5 s and ±15 s, distance of the fixes 5 s before and after to the arc) use only fixes already in the trip file. Same split, same folds, same seed. `PATHSNAP_FEATURES=v1` restores the old feature set.

| Model | CV Top-1 | CV macro F1 (std) | Student Top-1 (Review 1) | Student macro F1 | Student log loss | Train Top-1 | Fit time |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 89.1% | 0.9179 (0.012) | **87.3%** (86.5%) | 0.909 | 0.119 | 88.5% | 1.4 s |
| Gaussian Naive Bayes | 87.5% | 0.8325 (0.015) | **85.5%** (83.3%) | 0.823 | 0.905 | 87.1% | 0.1 s |
| HistGradientBoosting | 90.3% | 0.9402 (0.010) | **90.3%** (89.1%) | 0.937 | 0.088 | 91.8% | 6.4 s |
| Random Forest | 90.5% | 0.9412 (0.011) | **90.1%** (88.4%) | 0.937 | 0.083 | 96.2% | 139.9 s |
| MLP (64, 32) | 90.5% | 0.9370 (0.012) | **90.5%** (88.6%) | 0.937 | 0.086 | 91.7% | 65.2 s |
| HGB, Bayesian-tuned | 91.0% | 0.9415 (0.011) | **90.6%** (89.1%) | 0.938 | 0.083 | 96.7% | 18.6 s |

- Bayesian search re-run on the 26 features (40 trials, 78 min): best CV Top-1 91.0% (untuned 90.3%), student-test 90.6% (untuned 90.3%). The search only saw CV, so the student-test gain (+0.3 points) is the figure to trust; it is modest. Best values in `results/bayes_best_hgb.json`.
- Overfitting: the tuned model fits training rows harder (96.7% train vs 90.6% on unseen trips, a 6-point gap, like the forest); untuned boosting is at 91.8% vs 90.3%. Judge the tuned model on student-test, not on its training score.
- Boosting, forest and MLP are within about 0.6 points of each other on student-test, below the fold-to-fold spread, so their order is not firm.

| Student-test, tuned boosting | Review 1 | Final |
|---|---|---|
| Classifier only, Top-1 | 89.1% | 90.6% |
| Classifier + HMM, Top-1 (point-weighted) | 93.15% | 93.31% |
| Route continuity, classifier to HMM | 75.2% to 90.4% | 76.5% to 90.8% |
| Train out-of-fold Top-1, classifier to HMM | 89.4% to 92.3% | 91.0% to 93.1% |
| Route overlap IoU, classifier to HMM (labels ceiling 0.986) | 0.892 to 0.955 | 0.913 to 0.958 |
| Trips improved / worsened by the HMM | 10 / 0 | 10 / 0 |

Reading: the new features lift the classifier by about 1.5 points, and the HMM had already repaired much of that, so the end-to-end gain is smaller (+0.2 Top-1). Decoder settings were re-tuned on train out-of-fold probabilities only (lam 2, beta 2, gamma 0, penalty 20, unchanged). `scripts/check_predict.py` reproduces 93.3% through `predict_trip`.

## Review 1 results (17 features, `results/v1/`)

## Classifier comparison (`results/v1/model_runs.csv`)
| Model | Family | CV Top-1 | CV macro F1 (std) | Student Top-1 | Student macro F1 | Student log loss | Route continuity | Fit time | Inference |
|---|---|---|---|---|---|---|---|---|---|
| nearest + heading rule | rule baseline | 87.3% (train mean) | n/a | 86.0% | n/a | n/a | n/a | n/a | n/a |
| Logistic Regression | linear | 88.4% | 0.9149 (0.015) | 86.5% | 0.909 | 0.128 | 72.4% | 1.5 s | 0.17 µs/row |
| Gaussian Naive Bayes | probabilistic | 86.2% | 0.8390 (0.013) | 83.3% | 0.827 | 0.687 | 71.6% | 0.2 s | 0.41 µs/row |
| HistGradientBoosting | boosting | 89.2% | 0.9329 (0.014) | **89.1%** | 0.931 | 0.094 | 74.5% | 2.4 s | 1.1 µs/row |
| Random Forest | bagged trees | 89.4% | 0.9343 (0.014) | 88.4% | 0.929 | 0.094 | 74.4% | 50 s | 2.9 µs/row |
| MLP (64, 32) | neural network | 89.0% | 0.9294 (0.014) | 88.6% | 0.929 | 0.097 | 74.9% | 112 s | 1.1 µs/row |
| HGB, Bayesian-tuned | boosting | 89.5% | 0.9347 (0.014) | 89.1% | 0.931 | 0.092 | 75.2% | 8.4 s* | 3.2 µs/row* |

*Re-timed on an otherwise idle machine (best of 5): tuned HGB 8.4 s to fit and 3.2 µs per row to predict, untuned HGB 2.4 s and 0.95 µs per row. `results/model_runs.csv` holds 17 s and 13 µs for the tuned model because that run shared the CPU with other jobs. The tuned model has 103 leaves and 264 trees instead of 31 leaves and 100 trees, so it is about 3 times slower to predict.

The forest, boosting and MLP are within about a point of each other, and the fold-to-fold standard deviation of macro F1 is 0.014, so the ranking among them is not statistically solid (paired over the same folds: boosting minus forest is -0.18 points Top-1, boosting minus MLP +0.17). Boosting beats logistic regression in all 5 folds, by 0.8 points on average. The gap to the baseline (86.0% to about 89%) is real; logistic regression is only 0.5 points above the baseline and naive Bayes is below it.

## Overfitting and error evidence (`results/model_diagnostics.csv`, `confusion_matrices.csv`)
| Model | Train Top-1 | Student Top-1 | Train log loss | Student log loss | Student precision / recall |
|---|---|---|---|---|---|
| Logistic Regression | 87.9% | 86.5% | 0.125 | 0.128 | 0.841 / 0.853 |
| Gaussian NB | 85.9% | 83.3% | 0.580 | 0.687 | 0.608 / 0.888 |
| Random Forest | **95.1%** | 88.4% | 0.047 | 0.094 | 0.896 / 0.865 |
| HistGradientBoosting | 90.6% | 89.1% | 0.077 | 0.094 | 0.886 / 0.883 |
| MLP | 90.1% | 88.6% | 0.080 | 0.097 | 0.885 / 0.877 |

The forest overfits (6.7-point train-to-test gap). Logistic regression underfits (low on both). Naive Bayes is badly calibrated: many false positives, log loss 0.69, because the features are strongly correlated. On student-test, boosting's correct road is missing from the candidates for 49 of 19,157 labelled points (0.26%); of its remaining mistakes (10.6% of points), 32% pick the opposite direction of the correct road and 68% a different segment. The 68% splits into next arc on the same road, sibling at a junction and close parallel road, with about 3% elsewhere (`09_error_analysis.md`).

Figures: `report/figures/models/` (made by `scripts/make_model_plots.py`). Review scripts: `docs/presentation/`.

## Model choice and Bayesian optimisation
**Chosen family: HistGradientBoosting.** Best student-test Top-1 among the untuned models, the same CV quality as the forest, no overfitting gap like the forest's, and 20 to 45 times cheaper to train, which makes a Bayesian search affordable. Write your own justification in the report.

**Search** (`scripts/bayes_opt.py`, results in `results/bayes_trials_hgb.csv` and `bayes_best_hgb.json`): Optuna TPE, seed 0, 40 trials, about 34 minutes. Objective: mean observation-level Top-1 over the same 5 trip-grouped folds as the untuned run (Top-1 because it is the route-matching metric; log loss and macro F1 are logged for every trial). Search space: learning rate 0.02 to 0.3 (log), leaf nodes 8 to 128 (log), minimum samples per leaf 10 to 400 (log), L2 1e-4 to 10 (log), bins 64, 128 or 255, iterations 150 to 500. Trial 0 is the default-like configuration, so the search contains its own reference point. Best is trial 35: learning rate 0.024, 103 leaves, minimum leaf 19, L2 0.017, 255 bins, 264 iterations.

| | Untuned HGB | Tuned HGB |
|---|---|---|
| CV Top-1 (5 folds) | 89.20% | 89.48% |
| CV macro F1 | 0.9329 | 0.9347 |
| Student-test Top-1 | 89.14% | 89.09% |
| Student-test macro F1 | 0.9310 | 0.9314 |
| Student-test log loss | 0.0943 | 0.0920 |
| Fit time (idle machine) | 2.4 s | 8.4 s |

**Did tuning help?** Barely. CV Top-1 rises 0.28 points, the 40 trials span only 88.99% to 89.48% (median 89.37%), and the fold-to-fold standard deviation of Top-1 is about 2.6 points. Student-test Top-1 does not move (-0.05 points); only log loss improves a little. The default was already close to the optimum, and the report should say so rather than claim a big gain.

**Final model: tuned HGB + HMM**, chosen because it has the best CV Top-1, the best CV and student log loss, and the best Top-1 after HMM decoding (below). The difference to the untuned model is within noise, so either would be defensible. It is saved in `models/` (1.4 MB) and `predict_trip` uses it.

Epoch/iteration histories exist for HGB and the MLP only (`results/history_*.json`). The forest has none, so no training curve may be drawn for it. The HGB early-stopping hold-out and the MLP validation split use random rows, not whole trips, so they are for stopping only.

## HMM + Viterbi (`scripts/run_hmm.py`)
Decoder settings are tuned per classifier on train out-of-fold probabilities (the classifier never saw those trips). Grid: 50 combinations (emission weight 0.5 to 5, transition scale beta 1 to 10 m, gamma 0 or 0.1, flat penalty 20 for unreachable moves), in `results/hmm_grid_<model>.csv`. For both models the best setting lies strictly inside the grid and the surface is flat (best Top-1 per beta value within 0.2 points for beta 1 to 5), so the choice is not sensitive.

| | HGB untuned | HGB tuned (final) |
|---|---|---|
| Chosen setting | weight 1, beta 3, gamma 0 | weight 2, beta 2, gamma 0 |
| Train out-of-fold Top-1: classifier only to with HMM | 89.1% to 92.1% | 89.4% to **92.3%** |
| Student-test Top-1 (point-weighted): classifier only to with HMM | 89.1% to 92.8% | 89.1% to **93.2%** |
| Student-test route continuity (mean per trip): classifier only to with HMM | 74.5% to 90.9% | 75.2% to 90.4% |
| Student-test trips improved / worsened | 10 / 0 | 10 / 0 |

Per trip files: `results/hmm_student_test_hgb.csv`, `results/hmm_student_test_hgb_tuned.csv`. Student-test Top-1 improves on every trip for both models, by about 1 to 5 points on the untuned model. The weakest trip is still 77. HMM never made a student-test trip worse here, but the question "where did HMM not help" must be answered in the report from the train out-of-fold results too. `scripts/hmm_failure_analysis.py` is the starting point.

End to end check (`scripts/check_predict.py`): `predict_trip` run as the API will run it, on the 10 student-test trips, gives exactly 93.15% Top-1 and 90.4% route continuity, matching the numbers above, and takes at most about 0.9 s per trip on an idle machine (1.9 s when the CPU was shared).

## How the decoder works (so you can explain it)
- **Hidden states:** the candidate arcs of each GPS fix.
- **Emission:** classifier probability, normalised over that fix's candidates, as a log.
- **Transition:** minus the gap between the driven route distance (shortest path on the directed road graph from the previous candidate's position to this one's) and the straight-line distance between the two fixes, divided by beta. A move with no path within the search limit gets a flat penalty, not probability zero.
- **Gaps:** the chain restarts wherever `segment_ids` changes (time gap over 60 s).
- **Cost:** route distances are computed once per trip and cached; the tuning grid then only re-runs Viterbi.
- Code: `src/pathsnap/hmm.py`, tests in `tests/test_hmm.py`.

## Caveats
- Everything depends on our label alignment.
- The choice among the three strong models, and between untuned and tuned boosting, is within noise.
- Decoder and classifier were both tuned and evaluated on trips from the same region mix; evaluator trips may behave differently.
- The Bayesian search used 40 trials and ran while the CPU was shared, so its 34-minute duration is not a clean timing. Its objective is the CV score itself, so the CV gain (+0.28 points) is optimistic; the student-test result, which the search never saw, shows no gain.
- Where the HMM does not help (train out-of-fold, tuned model, `results/hmm_per_trip_hgb_tuned.csv`): better on 74 of 78 trips, worse on 1 (trip 72, minus 2.8 points), about equal on 3 (change within 0.1 points). Trip 72 is also the trip where the correct road is most often missing from the candidates (14.8%), and an HMM cannot choose a road that is not offered. Trip 16 stays at about 53% for both: 20% of its fixes are stopped and its errors are mostly parallel roads and direction flips (`09_error_analysis.md`).
