# Review 2 · Presentation script: hyperparameters, tuning, HMM, and the full system

About 12 minutes. Every number is from `results/` in this repo (see `docs/08_models_and_hmm.md`). Slides marked **[TO SHOW]** depend on work that is not finished yet (API, UI, Folium maps); do not describe them as done until they are.

AI-use note: drafted with AI help from our own results. We disclose that, and each of us must be able to explain every claim.

---

## Slide 1 · Where we left off (30 s)
**Say:** "In Review 1 we compared five classifier families and picked boosting. Today: what the hyperparameters of each model are, how we tuned the chosen one with Bayesian optimisation, how the HMM turns point-by-point guesses into a connected route, what the full system scores, and where it still fails."

## Slide 2 · Hyperparameters and loss of each model (2 min)
**Show:** the table.

| Model | Key hyperparameters (starting values) | Loss / objective |
|---|---|---|
| Logistic Regression | C = 1.0 (L2 penalty), max 500 iterations, features standardised | Log loss plus an L2 penalty |
| Gaussian Naive Bayes | variance smoothing 1e-9 | No iterative loss: fits a Gaussian per feature and class by maximum likelihood |
| Random Forest | 150 trees, at least 5 samples per leaf, square-root of features per split | Gini impurity to choose splits |
| HistGradientBoosting | learning rate 0.1, 31 leaves, up to 300 iterations, early stopping on a 10 percent hold-out, patience 15 | Binary log loss, fitted by gradient boosting |
| MLP | layers (64, 32), up to 60 epochs, batch 512, early stopping with patience 8; library defaults for Adam, ReLU and L2 penalty 1e-4 | Log loss minimised with Adam |

**Say:** "All five use the same data, folds and seed. Boosting stopped itself at about 100 of its 300 allowed trees, and the MLP stopped at epoch 54 of 60, so early stopping worked as intended."

## Slide 3 · Bayesian optimisation: the setup (1.5 min)
**Say:**
- "Boosting has several knobs that interact: learning rate, tree size, minimum leaf size, regularisation, and how many trees to use. Grid search would be expensive, and random search wastes trials. Bayesian optimisation learns from earlier trials where good settings live. We used Optuna's TPE sampler, which models the settings of good trials and bad trials separately and proposes new settings where good ones are more likely."
- "Objective: mean observation-level Top-1 over the same five trip-grouped folds we used for the untuned model. Top-1 because it is the route-matching metric. Log loss and macro F1 are logged for every trial."
- "Search space: learning rate 0.02 to 0.3, leaf nodes 8 to 128, minimum samples per leaf 10 to 400, L2 regularisation 1e-4 to 10, histogram bins 64, 128 or 255, and iterations 150 to 500. The first four are on a log scale."
- "40 trials, seed 0, about 34 minutes. Trial 0 is the default-like setting, so the search contains its own reference point. The evaluator data and student-test are never touched during the search."

## Slide 4 · Bayesian optimisation: what happened (2 min)
**Show:** the trial history (best CV Top-1 so far against trial number) and the best-trial row. *(A plot is not drawn yet. Koushik can make it from `results/bayes_trials_hgb.csv`.)*

**Say:**
- "The default setting scored 89.22 percent CV Top-1. The best, trial 35, scored 89.48 percent: learning rate 0.024, 103 leaves, minimum leaf 19, L2 0.017, 255 bins, 264 trees."
- "The curve rises fast and then flattens: 89.37 percent by trial 4, 89.41 by 19, 89.48 by trial 39. 36 of 40 trials beat the default, and 16 are within 0.1 points of the best. The worst trial, a high learning rate of 0.27 with only 22 leaves, got 88.99 percent."
- "Across the trials, more leaves and lower learning rates went with better scores (rank correlations of plus 0.71 and minus 0.34). Treat that as a hint, not proof, because the sampler concentrates trials near good regions."

**Tuned against untuned, same metrics and same partitions:**

| | Untuned | Tuned |
|---|---|---|
| CV Top-1 | 89.20% | 89.48% |
| CV macro F1 | 0.9329 | 0.9347 |
| Student-test Top-1 | 89.14% | 89.09% |
| Student-test macro F1 | 0.9310 | 0.9314 |
| Student-test log loss | 0.0943 | 0.0920 |
| Fit time (idle machine) | 2.4 s | 8.4 s |
| Inference per row | 0.95 µs | 3.2 µs |

**Say honestly:** "Did tuning help? Barely. CV Top-1 rose 0.28 points and tuned beat untuned in all five folds, but the search optimised the CV score itself, so that gain is optimistic. On student-test, which the search never saw, Top-1 did not change (89.14 against 89.09), and only log loss improved a little. The default was already close to the best. The tuned model is bigger and about three times slower to predict. We say that rather than claim a big gain."

## Slide 5 · Why the HMM (1 min)
**Say:** "The classifier sees one GPS point at a time. It can pick a road that is not connected to the previous point's road, or flip a road to its opposite direction. A real car drives along connected roads. The HMM adds that memory."

## Slide 6 · How the HMM works (2 min)
**Show:** a small diagram: two consecutive GPS points, candidate roads, the shortest path between them.

**Say, matching the report questions:**
1. **Hidden states.** "The candidate roads of each GPS point."
2. **Emission.** "The classifier's probability for each candidate, normalised across that point's candidates, as a log. A weight lets us trust the classifier more or less."
3. **Transition.** "The gap between the distance you would have to drive along the road network from the previous candidate to this one, and the straight-line distance between the two GPS points. A small gap is likely, a large gap is not. The network distance comes from shortest paths on the directed road graph."
4. **Missing candidates and large gaps.** "Points with no candidate are skipped and the next point is linked to the last one that had candidates. When the time between two fixes exceeds 60 seconds, or the implied speed is above 80 metres per second, the chain restarts, so we never invent a path across a long gap."
5. **Disconnected roads.** "If no path exists within a search limit, the move gets a flat penalty of 20 in log terms instead of probability zero, so one bad point cannot break the whole decode."
6. **Decoding.** "Viterbi in log space finds the most likely sequence."

## Slide 7 · Tuning the decoder (1 min)
**Say:** "The decoder has a few settings: the emission weight, the transition scale, an extra distance term, and the penalty. We tuned them on the training trips' out-of-fold probabilities, so the classifier had never seen those trips, over 50 combinations. For the final model the best was weight 2 and scale 2 metres. The result is flat: for transition scales 1 to 5 the best score varies by under 0.1 points, so we are not relying on a lucky setting. Student-test was used only to report."

## Slide 8 · Does the HMM help? (2 min)
**Show:** table and per-trip bars.

| Final model (tuned boosting) | Classifier only | Classifier + HMM |
|---|---|---|
| Train, out-of-fold Top-1 | 89.4% | 92.3% |
| Student-test Top-1 (point-weighted) | 89.1% | **93.2%** |
| Student-test route continuity (mean per trip) | 75.2% | **90.4%** |
| Student-test trips improved / worsened | n/a | 10 / 0 |

**Say:**
- "Top-1 rises about four points and route continuity rises 15 points. For reference, the aligned reference labels themselves score 93.2 percent on this continuity measure, because sampling can skip a road, so we are close to the practical ceiling."
- "Every student-test trip improves, between 0.8 and 7.3 points. The biggest gain is the hardest trip, 77: 79.1 percent to 86.4."
- **Where it did not help:** "On the 78 training trips, out of fold, the HMM helped on 74, was about equal on 3, and hurt on 1. The one it hurt is trip 72, minus 2.8 points, and it is also the trip where the correct road is most often missing from the candidates (14.8 percent). An HMM cannot choose a road that was never offered, and it can lock onto a consistent but wrong route. Trip 16 stays at about 53 percent either way, and we have not yet analysed why."

## Slide 9 · The final model and the full pipeline (1 min)
**Say:** "Final system: candidate generation, features, tuned boosting, then the HMM. The model and decoder settings are saved in the repository, 1.4 megabytes. We ran the prediction function exactly as the API will run it on the 10 student-test trips and got 93.15 percent Top-1 and 90.4 percent continuity, matching the offline numbers, at under a second per trip. The reference route is never read by the predictor."

## Slide 10 · API, UI and maps **[TO SHOW]** (1.5 min)
*(Sharath and Koushik. Only say what exists on the day.)*
- "API on Laptop 1: loads the model and road network and, for a trip ID, runs the whole pipeline. Ground-truth labels stay server-side, in a separate scoring module. Request logs are shown on that laptop." **[Sharath: not built yet]**
- "UI on Laptop 2: lists the ten evaluator trips, sends the chosen ID, and shows the predicted route and accuracy." **[Sharath: not built yet]**
- "Folium maps: raw GPS, road network, reference route, predicted route, with a legend and layer control." **[Koushik: not built yet]**

## Slide 11 · Limits and honesty (45 s)
**Say:**
- "All scores depend on our own label alignment. The organisers' labels may differ."
- "Tuned and untuned boosting, and the top three model families, are within noise of each other."
- "We quarantined two defective training trips and said so."
- "Evaluator trips may contain defects like those; the pipeline must not crash on them."
- "Next: failure analysis of trips 16 and 72, the API and UI, the report."

---

## Likely questions, with answers

**Why Bayesian optimisation and not grid search?** Boosting has interacting hyperparameters. A grid over six of them is huge, and Bayesian search spends trials near good regions. Of 40 trials, 36 beat the default.

**Why did you optimise Top-1, not log loss?** Top-1 is the route-matching metric. Log loss was logged for every trial and moved the same way: best trial 0.0933 against the default's 0.0970.

**Is the 89.48 percent CV score trustworthy?** It is optimistic: it is the score the search selected on. The student-test result, which the search never saw, is the honest check, and it shows no Top-1 gain.

**Why is the tuned model barely better?** The default was already near a flat optimum. 16 of 40 trials are within 0.1 points of the best.

**Which hyperparameters mattered?** Tree size and learning rate showed the clearest trend (more leaves and a lower rate went with better scores), but the effects are small.

**How does the HMM handle gaps?** It restarts the chain when the time gap exceeds 60 s or implied speed exceeds 80 m/s; points without candidates are skipped.

**What is the transition probability?** Minus the absolute gap between network driving distance and straight-line distance, divided by a scale of 2 metres, with a flat penalty if no path exists.

**What happens if two consecutive points are on disconnected roads?** The path search finds no route within its limit, the move is penalised by a flat 20, and Viterbi prefers a connected alternative if the classifier's probabilities allow.

**Did HMM always help?** No. On training trips it helped 74 of 78, was neutral on 3, and hurt 1 (trip 72), where the right road was often not in the candidate list.

**Did you use the evaluator data to tune?** No. The decoder was tuned on training out-of-fold probabilities and the search only on training CV; student-test was for reporting.

**Can every team member explain the code?** Everyone gives a walkthrough of their own part to the others before the event.
