# Review 1 · Presentation script: five models, why, and how accurate

About 10 minutes. Every number is from `results/` in this repo. Figures are in `report/figures/models/`. Read it, then say it in your own words: you will be asked about any line of it.

AI-use note: this script was drafted with AI help from our own results. Per the event rules we disclose that, and each of us must be able to explain every claim below.

---

## Slide 1 · The problem and what we built (45 s)
**Show:** the blue-dots-versus-green-line Folium map from the guide (or our own map of a trip).

**Say:** "GPS points are noisy, so a car on one road can look like it is on the next street. PathSnap reconstructs the road route from those points. Our design has two stages. A classifier scores which road segment each GPS point belongs to. Then an HMM with Viterbi decoding joins those choices into one connected route. Today I cover stage one: the five classifiers, why we chose them, and how accurate they are. The HMM and tuning are in Review 2."

## Slide 2 · What the data told us (1 min)
**Show:** a small table of the audit findings (`docs/02_data_audit.md`).

**Say:**
- "We have 90 trips: 80 for training and 10 for student testing. The final 10 evaluator trips we never touch."
- "The trips come from all over the world, and every trip has its own road network, so a road's id means nothing across trips. We therefore never use the id as a class. We score candidate roads using geometry."
- "Two things in the data forced design choices. First, about 95 percent of roads (median trip) have an opposite-direction twin at the same distance, so distance alone cannot tell the direction, and a nearest-road rule is right only about half the time (52 percent on student-test). Direction features are essential. Second, two training trips are defective: in trip 33 the reference route skips half the drive, and in trip 54 the road network is missing, so only 13 percent of its points are within 15 metres of any road. They are 12 percent of the training points, so we quarantined them and report that openly."

## Slide 3 · From GPS point to a learning problem (1.5 min)
**Show:** a diagram: one GPS point, the arcs within 40 m, the arrow of motion.

**Say:**
- "For every GPS point we take the 10 nearest road segments within 40 metres, and always add the opposite-direction twin. The correct road is missing from this list for only 0.6 percent of labelled points, so candidate generation is not the bottleneck."
- "The classifier answers one question for each point and candidate road: is this the reference road? That gives about 1.27 million rows."
- "Our features are: distance to the road, which side of it the point is on, position along it, road length, heading agreement between the car's motion and the road direction, speed, time since the last fix, rank among candidates, and the node degrees at both ends."
- "Training labels come from aligning each point to the reference route with a forward-only dynamic program that allows revisits, because the route file lists roads but not which point belongs to which."

## Slide 4 · How we evaluate fairly (1 min)
**Show:** a diagram of the partitions.

**Say:**
- "We fit only on the 78 clean training trips, capped at 1,000 points per trip so that one 19,000-point trip cannot dominate."
- "Cross-validation is grouped by trip: all points from one trip are always in the same fold. Otherwise neighbouring points would leak between training and validation and every score would be inflated. The three Moscow trips share roads, so they stay in one fold."
- "Student-test is for comparing models, never for fitting. Evaluator data is never used."
- "Our primary metric is observation-level Top-1: for each GPS point, is the highest-scoring candidate the reference road? We also report macro F1 and log loss on candidate rows, precision, recall, confusion matrices, and training and inference time. A simple rule, nearest road plus heading, is our yardstick: it gets 86.0 percent Top-1 on student-test."

## Slide 5 · The five models and why (3 min)
**Show:** the table below. One line each.

| # | Model | Family | Why we tried it | Main limit here |
|---|---|---|---|---|
| 1 | Logistic Regression | linear | Simple, fast, gives probabilities. Tells us how much a straight-line boundary can do | Cannot combine features non-linearly, for example distance matters differently when heading is unclear |
| 2 | Gaussian Naive Bayes | probabilistic | Cheapest generative model. Checks whether independence is a workable assumption | Our features are strongly correlated (distance, rank, gap to best), which breaks independence |
| 3 | Random Forest | ensemble of trees | Handles non-linear rules and mixed features, robust to scale | Deep trees can memorise training trips |
| 4 | HistGradientBoosting | boosting | Corrects errors step by step; strong on tabular data; fast | Needs regularisation and tuning |
| 5 | MLP (64, 32) | neural network | Learns feature interactions with a smooth output; gives a training curve | Needs scaling; slowest to train |

**Say:** "We deliberately chose five different families, because the point is to understand which kind of model suits this problem, not to run five versions of the same thing. The configurations are in `results/model_runs.csv`: for example the forest has 150 trees with at least 5 samples per leaf, boosting has 31 leaves and learning rate 0.1 with early stopping, and the MLP has two hidden layers of 64 and 32 units with early stopping."

## Slide 6 · Accuracy: the headline table (1.5 min)
**Show:** the table, then `cv_macro_f1.png` and `student_test_top1_macro_f1.png`.

| Model | CV Top-1 | CV macro F1 (std) | Student Top-1 | Student macro F1 | Student log loss | Train time | Inference |
|---|---|---|---|---|---|---|---|
| Rule baseline | 87.3% (train, mean per trip) | n/a | 86.0% | n/a | n/a | n/a | n/a |
| Logistic Regression | 88.4% | 0.915 (0.015) | 86.5% | 0.909 | 0.128 | 1.5 s | 0.17 µs/row |
| Gaussian NB | 86.2% | 0.839 (0.013) | 83.3% | 0.827 | 0.687 | 0.2 s | 0.41 µs/row |
| Random Forest | 89.4% | 0.934 (0.014) | 88.4% | 0.929 | 0.094 | 50 s | 2.9 µs/row |
| HistGradientBoosting | 89.2% | 0.933 (0.014) | **89.1%** | 0.931 | 0.094 | 2.4 s | 1.1 µs/row |
| MLP | 89.0% | 0.929 (0.014) | 88.6% | 0.929 | 0.097 | 112 s | 1.1 µs/row |

**Say:** "Three tree and neural models are clustered between 88.4 and 89.1 percent Top-1 on student-test. Logistic regression is only half a point above our hand-written rule, and Gaussian naive Bayes is below it."

## Slide 7 · What the numbers mean (2 min)
**Show:** `train_vs_student_top1.png`, `confusion_matrices.png`, `log_loss.png`.

**Say, in this order:**
1. **Overfitting.** "Random forest scores 95.1 percent Top-1 and log loss 0.047 on its own training rows, but 88.4 percent and 0.094 on unseen trips. That is a clear overfit. Boosting is 90.6 versus 89.1, and the MLP 90.1 versus 88.6: small gaps, so they generalise better. Logistic regression is 87.9 versus 86.5: a small gap but low on both, which points to underfitting, a model too simple for the problem."
2. **Naive Bayes fails on calibration.** "Its recall is good (0.89) but precision is only 0.61: it predicts 10,943 false positives against about 1,900 to 2,200 for the forest, boosting and MLP. Log loss is 0.69 against 0.09. The probabilities are overconfident because correlated features are counted as independent evidence. That matters because the HMM uses these probabilities."
3. **The top three are statistically tied.** "Across the same five folds, boosting minus forest is minus 0.18 points of Top-1 and boosting beats the MLP by 0.17, with fold-to-fold spread around 1.4 points of macro F1. We cannot honestly rank them on accuracy alone. Boosting does beat logistic regression in all five folds by about 0.8 points."
4. **Where errors are.** "On student-test the correct road is missing from the candidate list for only 49 of 19,157 labelled points, 0.26 percent. Boosting is wrong on 10.6 percent of the rest. Of those mistakes, about one third (650 of 2,032) pick the opposite direction of the right road, and about two thirds pick a different segment. Trip 77 is the hardest at 80 percent for boosting, and trips 7 and 52 reach about 98 percent." *(We have not yet analysed what the other two thirds are. Do not claim a cause.)*
5. **No leakage.** "We never selected using training accuracy; we compared on trip-grouped CV and on student-test."

## Slide 8 · Which model and why (45 s)
**Say:** "We chose HistGradientBoosting. Accuracy ties with the forest and the MLP, but boosting trains in 2.4 seconds against 50 for the forest and 112 for the MLP, it has the best student-test Top-1 and macro F1, it generalises better than the forest, and its speed makes a Bayesian hyperparameter search of 40 trials affordable. Review 2 covers that search and the HMM, where accuracy rises to about 92.8 percent."

---

## Likely questions, with answers

**Why not select the random forest, since it has the best CV score?** Its CV Top-1 is 0.18 points higher, well inside fold noise, it overfits (95.1 percent train versus 88.4 percent test), and it is 20 times slower to train. We chose on generalisation, speed and the tuning budget.

**What is macro F1 here, and why also Top-1?** Macro F1 is computed on candidate rows (reference versus not), averaged across both classes, so the 16 percent minority class counts as much as the majority. Top-1 is per GPS point and is what route matching needs, so it is our primary metric. The organisers have not defined macro F1, so this is our stated definition.

**Why is Top-1 only about 89 percent?** The classifier judges each GPS point alone and has no memory of where the car just was. Of boosting's student-test mistakes, about a third are a direction flip of the correct road. The HMM supplies the missing memory by requiring consecutive choices to be connected along the road network; with it Top-1 rises to about 92.8 percent.

**Why group the CV by trip?** Points in one trip are strongly correlated. Random splits would put neighbouring points in training and validation and inflate scores.

**Why exclude trips 33 and 54?** The reference route there is wrong or the network is missing (only 13 percent of trip 54's points are within 15 m of any road), so the labels would teach noise. We state this in the report.

**Did you use the evaluator data?** No. The code cannot even list that partition.

**Do the models have training curves?** Only boosting and the MLP; we did not draw curves for the forest, logistic regression or naive Bayes because they have no epochs.

**What would you do with more time?** Add trajectory context to the features, which the HMM partly provides.
