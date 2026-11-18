# 10 · Report draft: §3.2 (RF, HGB, MLP), §4, §5

DRAFT, prepared with AI assistance (disclose in the report as the event requires). The numbers come from `results/` and are reproducible. The wording is a starting point: each owner rewrites their section in their own words and must be able to explain every line in the viva. Do not paste it as it stands.

The organiser PDF numbers the §3.2 questions 1, 2, 5, 6, 7, 8, 9 (3 and 4 do not appear in the text). Answered below under those numbers.

Common setup for every number: 78 clean train trips, 5-fold CV grouped by trip, at most 1,000 points per trip, seed 0, student-test = 10 trips never used for fitting or tuning. Labels are our own alignment of the reference route.

---
## 3.2 Per-model answers

### Random Forest (bagged trees)
**1. Why suitable.** Every candidate road is a row of 17 numbers (distance, side, heading agreement, speed, rank, and similar). The right answer depends on combinations, for example a close road is right only if it also points the way the car is moving. Trees split on such combinations directly, need no scaling and handle the mix of metres, angles and counts.
**2. Strengths and limits.** Strengths: robust, few settings to get right, good CV (Top-1 89.4%, the highest mean among untuned models). Limits seen here: it memorises (below), it is the slowest tree model to fit (50.2 s) and to predict (2.9 µs per row), and it scores each fix alone, with no memory of the previous fix.
**5. Hyperparameters.** 150 trees, minimum 5 samples per leaf, features per split = square root of 17, bootstrap sampling (default), seed 0.
**6. Objective.** Each tree splits to reduce Gini impurity (scikit-learn default); the forest averages the trees' class votes into a probability.
**7 and 8. Student-test.** Top-1 88.4%, macro F1 0.929, log loss 0.094, route continuity 74.4%. CV: Top-1 89.4%, macro F1 0.934 (std 0.014).
**9. Over- or underfitting.** Overfits. Top-1 is 95.1% on the rows it trained on and 88.4% on unseen trips, a 6.7-point gap; log loss rises from 0.047 to 0.094. No other model shows a gap above 2.6 points (`results/model_diagnostics.csv`). Evidence is the gap, not a feeling. A forest has no epochs, so no training curve is drawn.

### HistGradient Boosting
**1. Why suitable.** Same tabular, interacting-feature reasoning as the forest, but trees are added one at a time and each fixes the previous trees' mistakes, which usually gives sharper probabilities. It is fast, which matters because the Bayesian search refits it many times.
**2. Strengths and limits.** Strengths: best student-test Top-1 (89.1%), small train-to-test gap, 2.4 s to fit, 1.1 µs per row. Limits: more settings to tune than the forest, and the same blind spot as every per-fix classifier: it cannot use where the car just was (about a third of its mistakes are direction flips, `09_error_analysis.md`).
**5. Hyperparameters (untuned run).** Up to 300 iterations (stopped at 100), learning rate 0.1, 31 leaf nodes per tree, early stopping with 10% hold-out and patience 15, scoring = loss. The tuned values are in §4.
**6. Objective.** Binary log loss (cross-entropy), minimised stage by stage by gradient boosting.
**7 and 8. Student-test.** Top-1 89.1%, macro F1 0.931, log loss 0.094, route continuity 74.5%. CV: Top-1 89.2%, macro F1 0.933 (std 0.014).
**9. Over- or underfitting.** Neither strongly. 90.6% on training rows against 89.1% on unseen trips (1.4 points). The loss curve (`report/figures/models/training_curves.png`) shows training and hold-out loss falling together and flattening around iteration 80 to 100, with early stopping ending the run. That hold-out is random rows, not whole trips, so it is only for stopping, not an honest test.

### MLP (64, 32)
**1. Why suitable.** A small neural network can learn its own mixtures of the 17 features. It is the neural-network family for the comparison; with only 17 numeric inputs it should compete with trees if the problem is smooth.
**2. Strengths and limits.** Strengths: smooth probabilities, 1.1 µs per row, accuracy level with the forest. Limits: needs scaled inputs (we add a standard scaler fitted on training rows only), 111.6 s to fit, and no gain over boosting for that cost.
**5. Hyperparameters.** Two hidden layers (64 and 32 units), early stopping with 10% hold-out and patience 8, up to 60 epochs (stopped after 54), batch size 512, Adam optimiser and ReLU (scikit-learn defaults), seed 0.
**6. Objective.** Binary log loss, minimised with Adam on mini-batches.
**7 and 8. Student-test.** Top-1 88.6%, macro F1 0.929, log loss 0.097, route continuity 74.9%. CV: Top-1 89.0%, macro F1 0.929 (std 0.014).
**9. Over- or underfitting.** Mild overfit at most: 90.1% training against 88.6% on unseen trips (1.5 points). The training-loss curve falls and flattens over 54 epochs; the hold-out accuracy (random rows) is noisy at the third decimal. See the same figure.

### Across the three
The three are within about a point of each other on student-test and the fold-to-fold standard deviation of macro F1 is 0.014, so their ranking is not statistically firm. Paired over the same folds, boosting minus forest is -0.18 points Top-1 and boosting minus MLP +0.17. What is solid: all three beat the rule baseline (86.0%) and logistic regression (86.5%) and naive Bayes (83.3%).

---
## 4 Bayesian optimisation (HistGradient Boosting)
**1. Which hyperparameters.** Learning rate, maximum leaf nodes, minimum samples per leaf, L2 regularisation, number of histogram bins, number of iterations. Early stopping stayed on in every trial (10% hold-out, patience 15), so the iteration count is an upper limit.
**2. Ranges.** Learning rate 0.02 to 0.3 (log scale), leaf nodes 8 to 128 (log, integer), minimum leaf 10 to 400 (log, integer), L2 1e-4 to 10 (log), bins 64, 128 or 255, iterations 150 to 500 (integer).
**3. Objective.** Mean observation-level Top-1 over the same 5 trip-grouped folds as the untuned run (the route-matching metric); log loss and macro F1 were logged for every trial. Optuna TPE sampler, seed 0.
**4. Trials.** 40, about 34 minutes. Trial 0 is the default-like configuration, so the search contains its own reference point. Log: `results/bayes_trials_hgb.csv`.
**5. Best values.** Trial 35: learning rate 0.024, 103 leaves, minimum leaf 19, L2 0.017, 255 bins, 264 iterations (`results/bayes_best_hgb.json`).
**6. Tuned vs untuned.**

| | Untuned | Tuned |
|---|---|---|
| CV Top-1 | 89.20% | 89.48% |
| CV macro F1 | 0.9329 | 0.9347 |
| Student-test Top-1 | 89.14% | 89.09% |
| Student-test macro F1 | 0.9310 | 0.9314 |
| Student-test log loss | 0.0943 | 0.0920 |
| Fit time (idle machine) | 2.4 s | 8.4 s |

Tuning helped barely. CV Top-1 rose 0.28 points, but the 40 trials span only 88.99% to 89.48% and the fold-to-fold standard deviation is about 2.6 points. The search optimised the CV score itself, so that gain is optimistic. On student-test, which the search never saw, Top-1 did not move (-0.05 points) and only log loss improved a little. Say this plainly rather than claiming a large gain. We kept the tuned model because it has the best CV Top-1 and log loss; the difference to the untuned model is within noise.

---
## 5 HMM and Viterbi
**1. Hidden states.** For each GPS fix, the candidate road arcs near it (10 nearest within 40 m, widened to 80 m if none, plus the opposite-direction twin of each).
**2. Emissions.** The classifier's probability for each candidate, normalised over that fix's candidates, taken as a log and multiplied by a weight (lam = 2).
**3. Transitions.** Between consecutive fixes, the shortest driven distance on the directed road graph from the previous candidate's position to this one's, compared with the straight-line distance between the two fixes. The score is minus the absolute difference divided by beta (beta = 2 m, gamma = 0). A candidate pair whose driven distance matches the GPS distance is likely; one that needs a long detour is not.
**4. Missing candidates and gaps.** A time gap over 60 s or an implied speed over 80 m/s starts a new segment and the chain restarts there. A fix with no arc within 80 m is left out of the chain and gets no arc. If the reference road is not among a fix's candidates (0.26% of student-test points for untuned boosting) the HMM cannot choose it.
**5. Disconnected moves.** If no path exists within the search limit, the move gets a flat penalty of 20 rather than probability zero, so decoding never fails; the decoder then prefers a connected alternative when one exists.
**6. Did it improve the route?** Yes, on every student-test trip.

| Student-test (tuned boosting) | Classifier only | Classifier + HMM |
|---|---|---|
| Top-1 | 89.1% | 93.2% |
| Route continuity (mean per trip) | 75.2% | 90.4% |
| Route overlap IoU, direction counted | 0.892 | 0.955 |
| Trips improved / worsened | | 10 / 0 |

References for the ceiling: the aligned labels themselves score 93.2% continuity and 0.986 overlap, so the HMM is close to what one-arc-per-fix prediction can reach. Decoder settings were tuned on train out-of-fold probabilities only; train Top-1 went 89.4% to 92.3%.

Where it did not help (train out-of-fold, `results/hmm_per_trip_hgb_tuned.csv`): better on 74 of 78 trips, about equal on 3, worse on 1 (trip 72, minus 2.8 points). Trip 72 is the trip where the correct road is most often missing from the candidates (14.8%), and an HMM cannot choose a road that is not offered. Trip 16 stays near 53% with or without the HMM: 20% of its fixes are stopped and its mistakes are mostly parallel roads (`09_error_analysis.md`).

Limits to state: all numbers depend on our label alignment; classifier and decoder were tuned and tested on the same region mix, so evaluator trips may behave differently.
