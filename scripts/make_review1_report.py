"""Short Review 1 handout: why each model, and its accuracy. Numbers come from results/*.csv.

    python scripts/make_review1_report.py RESULTS_DIR FIG_DIR OUT_PDF
"""
import os
import sys

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

RES, FIG, OUT = sys.argv[1:4]
runs = pd.read_csv(os.path.join(RES, "model_runs.csv")).set_index("model")
diag = pd.read_csv(os.path.join(RES, "model_diagnostics.csv")).set_index("model")
base = pd.read_csv(os.path.join(RES, "baseline_student_test.csv"))
b_top1 = float((base.top1_with_heading * base.labelled_points).sum() / base.labelled_points.sum())

ss = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=17, spaceAfter=6)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12.5, spaceBefore=10, spaceAfter=4)
P = ParagraphStyle("p", parent=ss["BodyText"], fontSize=10, leading=13.5)
SM = ParagraphStyle("sm", parent=P, fontSize=8.5, leading=11, textColor=colors.HexColor("#52514e"))
CELL = ParagraphStyle("cell", parent=P, fontSize=8.3, leading=10.3)

ORDER = ["lr", "gnb", "rf", "hgb", "mlp"]
NAME = dict(lr="Logistic Regression", gnb="Gaussian Naive Bayes", rf="Random Forest", hgb="HistGradient Boosting", mlp="MLP (64, 32)")
WHY = dict(
    lr="Simple linear model. Shows how far the 17 features get with one straight decision boundary, and gives a cheap, readable reference.",
    gnb="Probabilistic model that assumes the features are independent. Included as a deliberately weak, very fast family to compare against.",
    rf="Many decision trees averaged. Handles curved, interacting features (distance with heading) without scaling, and is hard to beat on tabular data.",
    hgb="Trees built one after another, each fixing the last one's mistakes. Same strengths as the forest, much faster to train, so a Bayesian search is affordable.",
    mlp="Small neural network. Tests whether learned feature mixing beats trees on 17 numeric inputs.",
)
pct = lambda v: f"{v * 100:.1f}%"


def img(name, width):
    from reportlab.lib.utils import ImageReader
    w, h = ImageReader(os.path.join(FIG, name)).getSize()
    return Image(os.path.join(FIG, name), width=width, height=width * h / w)


def table(data, widths, head=True):
    t = Table(data, colWidths=widths, repeatRows=1)
    st = [("FONTSIZE", (0, 0), (-1, -1), 8.3), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c3c2b7")),
          ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f1fc")), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(st))
    return t


s = []
s += [Paragraph("PathSnap: Review 1 handout", H1),
      Paragraph("Map-matching GPS fixes to roads. UE24CS352A. Team 1, Section C. Student-test and train data only; evaluator data not used.", SM),
      Paragraph("1. What we built", H2),
      Paragraph("Each GPS fix has about 6.5 candidate road arcs nearby on average (the 10 nearest within 40 m, plus the opposite-direction twin of each). "
                "A classifier scores every candidate with 17 features (distance, side of road, heading agreement, speed, rank and similar) "
                "and answers one question: is this the reference road? The highest score is the prediction. "
                "A rule baseline (nearest road plus a heading check) reaches " + pct(b_top1) + " on the student-test trips, "
                "so any model has to beat that to be worth using. The HMM/Viterbi step that joins the choices into a route is Review 2.", P),
      Paragraph("2. How we measured", H2),
      Paragraph("80 train trips; 2 were quarantined automatically (trips 33 and 54 have broken references), so we fit on 78. "
                "Cross-validation is 5-fold grouped by trip, so a trip is never split between fitting and testing. "
                "At most 1,000 points per trip are used for fitting, so one long trip cannot dominate. "
                "The 10 student-test trips were never used for fitting or tuning and give the second set of numbers. "
                "Top-1 means the share of GPS fixes where the best-scored road is the reference road. "
                "Macro F1 is computed over all candidate rows. Labels are our own alignment of the supplied route, so absolute values may shift with the organisers' labels.", P),
      Paragraph("3. Results: all five models", H2)]

hdr = [Paragraph(f"<b>{h}</b>", CELL) for h in ["Model", "CV Top-1", "CV macro F1 (std)", "Student Top-1", "Student macro F1", "Student log loss", "Fit time"]]
rows = [hdr]
for m in ORDER:
    r = runs.loc[m]
    sl = "" if pd.isna(r.student_log_loss) else f"{r.student_log_loss:.3f}"
    rows.append([NAME[m], pct(r.cv_top1_mean), f"{r.cv_macro_f1_mean:.3f} ({r.cv_macro_f1_std:.3f})", pct(r.student_top1),
                 f"{r.student_macro_f1:.3f}", sl, f"{r.final_fit_seconds:g} s"])
rows.append(["Rule baseline (no learning)", "n/a", "n/a", pct(b_top1), "n/a", "n/a", "n/a"])
s += [table(rows, [4.3 * cm, 1.9 * cm, 3.0 * cm, 2.1 * cm, 2.2 * cm, 2.0 * cm, 1.7 * cm]), Spacer(1, 4),
      Paragraph("Source: results/model_runs.csv. Seed 0, same folds for every model.", SM),
      Spacer(1, 6), img("11_cv_by_fold.png", 16 * cm), Spacer(1, 4), img("12_student_test_by_trip.png", 13 * cm)]

s += [Paragraph("4. Why each model, and what we saw", H2)]
SEEN = dict(
    lr=f"Student Top-1 {pct(runs.loc['lr','student_top1'])}, only about half a point above the rule. It underfits: low on both train ({pct(diag.loc['lr','train_top1'])}) and test. A straight boundary cannot express 'close and pointing the right way'.",
    gnb=f"Student Top-1 {pct(runs.loc['gnb','student_top1'])}, below the rule. The features are strongly correlated, which breaks its independence assumption, so it is over-confident (log loss {runs.loc['gnb','student_log_loss']:.2f}, five to seven times the others).",
    rf=f"Student Top-1 {pct(runs.loc['rf','student_top1'])} and the best CV Top-1 ({pct(runs.loc['rf','cv_top1_mean'])}), but it overfits: {pct(diag.loc['rf','train_top1'])} on train rows against {pct(diag.loc['rf','student_top1'])} on unseen trips. It also takes {runs.loc['rf','final_fit_seconds']:g} s to fit.",
    hgb=f"Best student Top-1 ({pct(runs.loc['hgb','student_top1'])}) with a small train-to-test gap ({(diag.loc['hgb','train_top1']-diag.loc['hgb','student_top1'])*100:.1f} points) and only {runs.loc['hgb','final_fit_seconds']:g} s to fit.",
    mlp=f"Student Top-1 {pct(runs.loc['mlp','student_top1'])}, level with the forest and boosting, but {runs.loc['mlp','final_fit_seconds']:g} s to fit and no accuracy gain over boosting.",
)
for m in ORDER:
    s += [Paragraph(f"<b>{NAME[m]}.</b> <i>Why:</i> {WHY[m]} <i>Result:</i> {SEEN[m]}", P), Spacer(1, 4)]

s += [Paragraph("5. Which model and why", H2),
      Paragraph("We carry <b>HistGradient Boosting</b> forward. It has the best student-test Top-1, no forest-style overfitting gap, and trains in "
                f"{runs.loc['hgb','final_fit_seconds']:g} s against {runs.loc['rf','final_fit_seconds']:g} s (forest) and {runs.loc['mlp','final_fit_seconds']:g} s (MLP), "
                "which makes a Bayesian hyperparameter search affordable in Review 2. We state the limit plainly: forest, boosting and MLP are within about one point of each other, "
                "and fold-to-fold spread (about 0.014 macro F1) is larger than those gaps, so the ranking among the three is not statistically solid. "
                "The gap that is real is from the rule (" + pct(b_top1) + ") to about 89%, and from naive Bayes and logistic regression to the rest.", P),
      Spacer(1, 4), img("13_time_vs_accuracy.png", 12.5 * cm), Spacer(1, 4), img("14_overfitting.png", 12 * cm),
      Paragraph("6. Known gaps", H2),
      Paragraph("About a third of boosting's mistakes (650 of 2,081 student-test fixes) are the opposite direction of the right road; two thirds pick a different road segment, which we have not analysed yet. "
                "Route-level accuracy and the HMM decoder come in Review 2. Draft prepared with AI assistance as allowed by the event rules; team members rewrite and own the final text.", P)]

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                        title="PathSnap Review 1 handout", author="Team 1, Section C")
doc.build(s)
print("wrote", OUT)
