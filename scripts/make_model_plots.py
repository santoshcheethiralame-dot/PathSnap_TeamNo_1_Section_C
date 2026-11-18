"""Model-comparison figures, drawn only from files in results/.

    python scripts/make_model_plots.py RESULTS_DIR OUT_DIR

No curve is drawn for a model that has no epoch history.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ORDER = ["lr", "gnb", "rf", "hgb", "mlp"]
LABEL = {"lr": "Logistic\nRegression", "gnb": "Gaussian\nNaive Bayes", "rf": "Random\nForest",
         "hgb": "HistGradient\nBoosting", "mlp": "MLP", "hgb_tuned": "HGB\n(tuned)"}


def main():
    res, out = sys.argv[1:3]
    os.makedirs(out, exist_ok=True)
    runs = pd.read_csv(os.path.join(res, "model_runs.csv")).set_index("model")
    folds = pd.read_csv(os.path.join(res, "cv_folds.csv"))
    diag = pd.read_csv(os.path.join(res, "model_diagnostics.csv")).set_index("model")
    cm = pd.read_csv(os.path.join(res, "confusion_matrices.csv")).set_index("model")
    base = pd.read_csv(os.path.join(res, "baseline_student_test.csv"), dtype={"trip": str})
    base_top1 = float((base.top1_with_heading * base.labelled_points).sum() / base.labelled_points.sum())
    models = [m for m in ORDER if m in runs.index]
    x = np.arange(len(models))
    names = [LABEL[m] for m in models]

    def save(name):
        plt.tight_layout()
        plt.savefig(os.path.join(out, name), dpi=160)
        plt.close()

    # 1. CV macro F1 with fold spread
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, m in enumerate(models):
        v = folds[folds.model == m].macro_f1
        ax.bar(i, v.mean(), yerr=v.std(), capsize=4, color="#4c72b0")
        ax.scatter([i] * len(v), v, color="k", s=10, zorder=3)
    ax.set_xticks(x, names)
    ax.set_ylim(0.8, 0.96)
    ax.set_ylabel("Macro F1 (5 trip-grouped folds)")
    ax.set_title("Cross-validation macro F1 (train trips)")
    save("cv_macro_f1.png")

    # 2. student-test Top-1 and macro F1
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].bar(x, runs.loc[models, "student_top1"], color="#55a868")
    axs[0].axhline(base_top1, color="r", ls="--", label=f"rule baseline {base_top1:.3f}")
    axs[0].set_ylim(0.8, 0.92); axs[0].set_xticks(x, names); axs[0].legend(); axs[0].set_title("Student-test Top-1")
    axs[1].bar(x, runs.loc[models, "student_macro_f1"], color="#55a868")
    axs[1].set_ylim(0.8, 0.95); axs[1].set_xticks(x, names); axs[1].set_title("Student-test macro F1")
    save("student_test_top1_macro_f1.png")

    # 3. log loss
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.38
    ax.bar(x - w / 2, runs.loc[models, "cv_log_loss_mean"], w, label="CV", color="#4c72b0")
    ax.bar(x + w / 2, runs.loc[models, "student_log_loss"], w, label="student-test", color="#55a868")
    ax.set_yscale("log"); ax.set_xticks(x, names); ax.set_ylabel("Log loss (log scale)"); ax.legend()
    ax.set_title("Log loss: lower is better")
    save("log_loss.png")

    # 4. train vs student-test Top-1 (overfitting evidence)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, diag.loc[models, "train_top1"], w, label="training rows", color="#c44e52")
    ax.bar(x + w / 2, diag.loc[models, "student_top1"], w, label="student-test", color="#55a868")
    ax.set_ylim(0.8, 0.97); ax.set_xticks(x, names); ax.set_ylabel("Top-1"); ax.legend()
    ax.set_title("Train vs student-test Top-1")
    save("train_vs_student_top1.png")

    # 5. confusion matrices (student-test, candidate level, threshold 0.5)
    fig, axs = plt.subplots(1, len(models), figsize=(3 * len(models), 3.2))
    for ax, m in zip(axs, models):
        c = cm.loc[m]
        mat = np.array([[c.tn, c.fp], [c.fn, c.tp]])
        ax.imshow(mat / mat.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(mat):
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=8)
        ax.set_xticks([0, 1], ["pred 0", "pred 1"]); ax.set_yticks([0, 1], ["true 0", "true 1"])
        ax.set_title(LABEL[m].replace("\n", " "), fontsize=9)
    save("confusion_matrices.png")

    # 6. training histories, only for models that have them
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    p = os.path.join(res, "history_hgb.json")
    if os.path.exists(p):
        h = json.load(open(p))
        axs[0].plot(-np.array(h["train_score"]), label="train (log loss)")
        axs[0].plot(-np.array(h["val_score"]), label="hold-out (log loss)")
        axs[0].set_title("HistGradientBoosting"); axs[0].set_xlabel("iteration"); axs[0].legend()
    p = os.path.join(res, "history_mlp.json")
    if os.path.exists(p):
        h = json.load(open(p))
        axs[1].plot(h["train_loss"], label="train loss")
        ax2 = axs[1].twinx()
        ax2.plot(h["val_score"], color="C1", label="hold-out accuracy")
        axs[1].set_title("MLP"); axs[1].set_xlabel("epoch"); axs[1].legend(loc="upper center"); ax2.legend(loc="center right")
    save("training_curves.png")
    print("wrote figures to", out)


if __name__ == "__main__":
    main()
