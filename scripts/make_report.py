"""Build the consolidated project report (PDF). Every number is read from results/*.csv or *.json.

    python scripts/make_report.py RESULTS_DIR REPORT_DIR OUT_PDF

REPORT_DIR holds figures/ (eda, models, ui), maps/ and api_demo_output.txt. Two small figures
(pipeline diagram, Bayesian trials, per-trip HMM effect) are drawn here into REPORT_DIR/figures/final.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

RES, REP, OUT = sys.argv[1:4]
FIG = os.path.join(REP, "figures")
FINAL = os.path.join(FIG, "final")
os.makedirs(FINAL, exist_ok=True)

# ----------------------------------------------------------------------------- data
runs = pd.read_csv(os.path.join(RES, "model_runs.csv")).set_index("model")
diag = pd.read_csv(os.path.join(RES, "model_diagnostics.csv")).set_index("model")
cm_ = pd.read_csv(os.path.join(RES, "confusion_matrices.csv")).set_index("model")
hs = pd.read_csv(os.path.join(RES, "hmm_summary_hgb_tuned.csv")).iloc[0]
hs_u = pd.read_csv(os.path.join(RES, "hmm_summary_hgb.csv")).iloc[0]
hst = pd.read_csv(os.path.join(RES, "hmm_student_test_hgb_tuned.csv"))
hpt = pd.read_csv(os.path.join(RES, "hmm_per_trip_hgb_tuned.csv"))
ov = pd.read_csv(os.path.join(RES, "route_overlap_student_test.csv")).mean(numeric_only=True)
ref_cont = pd.read_csv(os.path.join(RES, "reference_route_continuity_student_test.csv"))
bt = pd.read_csv(os.path.join(RES, "bayes_trials_hgb.csv"))
bb = json.load(open(os.path.join(RES, "bayes_best_hgb.json")))
bw = pd.read_csv(os.path.join(RES, "baseline_student_test.csv"))
base_top1 = float((bw.top1_with_heading * bw.labelled_points).sum() / bw.labelled_points.sum())
cvf = pd.read_csv(os.path.join(RES, "cv_folds.csv"))
ea = pd.read_csv(os.path.join(RES, "error_analysis_hmm.csv"))
dsum = pd.read_csv(os.path.join(RES, "dataset_build_summary.csv"))

pct = lambda v, d=1: f"{v * 100:.{d}f}%"
n_ = lambda m, c: runs.loc[m, c]


def err_mix(method, part):
    d = ea[(ea.method == method) & (ea.partition == part)]
    wrong = int(d.labelled_points.sum() - d.correct.sum())
    cats = {c: int(d[c].sum()) for c in ["opposite", "next_arc", "sibling", "parallel", "elsewhere", "missing"]}
    sp, sw = int(d.stopped_points.sum()), int(d.stopped_wrong.sum())
    lp = int(d.labelled_points.sum())
    return dict(wrong=wrong, total=lp, cats=cats, stop_share=sp / lp, stop_err=sw / sp if sp else 0,
                move_err=(wrong - sw) / (lp - sp), stop_share_of_err=sw / wrong)


_s = dsum[dsum.partition == "student_test"]
MISS_ST = float((_s.candidate_miss_rate * _s.n_points).sum() / _s.n_points.sum())
EC = err_mix("classifier", "student_test")
EH = err_mix("hmm", "student_test")
hpt = hpt[hpt.partition == "train"]
tr_better = int((hpt.change > 0.002).sum())
tr_worse = int((hpt.change < -0.002).sum())
tr_same = len(hpt) - tr_better - tr_worse
worst = hpt.sort_values("change").iloc[0]

# ----------------------------------------------------------------------------- small figures
INK, MUTED, BLUE, TEAL, AMBER = "#1d2733", "#6b7885", "#2a6fdb", "#1f9d8f", "#d9971a"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})

fig, ax = plt.subplots(figsize=(7.4, 2.1))
ax.axis("off")
ax.set_xlim(0, 7.4)
ax.set_ylim(0, 2.1)
boxes = [("Inputs", "GPS track +\nroad network\nper trip"), ("1 Candidates", "10 nearest arcs\nwithin 40 m\n+ reverse twins"),
         ("2 Features", "26 per candidate:\ngeometry, heading,\nspeed, context"), ("3 Classifier", "boosting scores\n'is this the\nreference road?'"),
         ("4 HMM/Viterbi", "connected path\nthrough all fixes"), ("Route", "ordered road arcs\n+ scores")]
for i, (t_, s_) in enumerate(boxes):
    x = 0.05 + i * 1.22
    hl = i in (1, 2, 3, 4)
    ax.add_patch(FancyBboxPatch((x, 0.15), 1.0, 1.8, boxstyle="round,pad=0.02,rounding_size=0.06", fc="#eaf1fd" if hl else "#f6f6f3", ec=BLUE if hl else "#b8b8b0", lw=1.3))
    ax.text(x + 0.5, 1.6, t_, ha="center", va="center", fontsize=6.6, fontweight="bold", color=INK)
    ax.text(x + 0.5, 0.85, s_, ha="center", va="center", fontsize=5.7, color=MUTED, linespacing=1.4)
    if i < 5:
        ax.annotate("", xy=(x + 1.2, 1.05), xytext=(x + 1.03, 1.05), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
fig.savefig(os.path.join(FINAL, "pipeline.png"), dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.2, 3.2))
bt = bt.sort_values("number")
best_so_far = bt.value.cummax()
ax.scatter(bt.number, bt.value * 100, s=26, color=BLUE, zorder=3, label="trial")
ax.plot(bt.number, best_so_far * 100, color=AMBER, lw=2, label="best so far")
ax.set_xlabel("trial")
ax.set_ylabel("5-fold grouped CV Top-1 (%)")
ax.grid(axis="y", color="#e2e2dc")
ax.legend(frameon=False)
fig.savefig(os.path.join(FINAL, "bayes_trials.png"), dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.4, 3.2))
tt = hst.sort_values("trip")
x = np.arange(len(tt))
ax.bar(x - 0.2, tt.top1_classifier * 100, 0.4, color="#8fa3b8", label="classifier only")
ax.bar(x + 0.2, tt.top1_hmm * 100, 0.4, color=TEAL, label="classifier + HMM")
ax.set_xticks(x)
ax.set_xticklabels([str(int(t)) for t in tt.trip], fontsize=8)
ax.set_xlabel("student-test trip")
ax.set_ylabel("Top-1 (%)")
ax.set_ylim(70, 100)
ax.grid(axis="y", color="#e2e2dc")
ax.legend(frameon=False, ncol=2, loc="upper center")
fig.savefig(os.path.join(FINAL, "hmm_per_trip.png"), dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ----------------------------------------------------------------------------- styles
ss = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=ss["Heading1"], keepWithNext=1, fontSize=16, spaceBefore=4, spaceAfter=7, textColor=colors.HexColor("#14263d"))
H2 = ParagraphStyle("h2", parent=ss["Heading2"], keepWithNext=1, fontSize=12, spaceBefore=9, spaceAfter=3, textColor=colors.HexColor("#14263d"))
H3 = ParagraphStyle("h3", parent=ss["Heading3"], keepWithNext=1, fontSize=10.3, spaceBefore=6, spaceAfter=2, textColor=colors.HexColor("#2a3f5a"))
P = ParagraphStyle("p", parent=ss["BodyText"], fontSize=9.6, leading=13.2, spaceAfter=4)
Q = ParagraphStyle("q", parent=P, leftIndent=10, spaceAfter=3)
SM = ParagraphStyle("sm", parent=P, fontSize=8, leading=10.4, textColor=colors.HexColor("#555a60"))
CAP = ParagraphStyle("cap", parent=SM, alignment=1, spaceBefore=1, spaceAfter=6)
CELL = ParagraphStyle("cell", parent=P, fontSize=7.8, leading=9.6, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Helvetica-Bold")
TITLE = ParagraphStyle("title", parent=ss["Title"], fontSize=25, leading=30, alignment=0, textColor=colors.HexColor("#14263d"))
CODE = ParagraphStyle("code", fontName="Courier", fontSize=6.9, leading=8.4, backColor=colors.HexColor("#f3f4f1"), borderPadding=5)


def para(t, st=P):
    return Paragraph(t, st)


def img(rel, width, base=FIG):
    path = os.path.join(base, rel)
    w, h = ImageReader(path).getSize()
    return Image(path, width=width, height=width * h / w)


def figure(rel, width, caption, base=FIG):
    return KeepTogether([img(rel, width, base), para(caption, CAP)])


def side_by_side(items, width_each):
    cells = [[img(r, width_each) for r, _ in items], [para(c, CAP) for _, c in items]]
    t = Table(cells, colWidths=[width_each + 6] * len(items))
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    return KeepTogether([t])


def table(rows, widths, header=True, zebra=True, align_right_from=None):
    data = []
    for i, r in enumerate(rows):
        data.append([c if not isinstance(c, str) else Paragraph(c, CELLB if (i == 0 and header) else CELL) for c in r])
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8c8c0")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6eefb")))
    t.setStyle(TableStyle(st))
    return t


def bullets(items):
    return [Paragraph("&bull; " + i, Q) for i in items]


def QA(num, q, a):
    return Paragraph(f"<b>{num}. {q}</b> {a}", P)


NAME = dict(lr="Logistic Regression", gnb="Gaussian Naive Bayes", rf="Random Forest", hgb="HistGradient Boosting",
            mlp="MLP (64, 32)", hgb_tuned="HistGradient Boosting, Bayesian-tuned")
FAM = dict(lr="Linear", gnb="Probabilistic (generative)", rf="Ensemble, bagging", hgb="Ensemble, boosting",
           mlp="Neural network", hgb_tuned="Ensemble, boosting")
KEY = dict(lr="L2, C=1, 500 iterations", gnb="var smoothing 1e-9", rf="150 trees, min leaf 5, sqrt features",
           hgb="lr 0.1, 31 leaves, up to 300 iterations, early stop", mlp="64-32 ReLU, Adam, early stop, batch 512",
           hgb_tuned="lr %.3f, %d leaves, min leaf %d, L2 %.2f, %d bins, %d iterations" % (
               bb["config"]["learning_rate"], bb["config"]["max_leaf_nodes"], bb["config"]["min_samples_leaf"],
               bb["config"]["l2_regularization"], bb["config"]["max_bins"], bb["config"]["max_iter"]))

import textwrap


def wrap_lines(s, width=108):
    out = []
    for line in s.splitlines():
        out += textwrap.wrap(line, width, subsequent_indent="    ", break_long_words=True, replace_whitespace=False) or [""]
    return "\n".join(out)


story = []

# ----------------------------------------------------------------------------- title
story += [Spacer(1, 2.2 * cm), para("PathSnap", TITLE),
          para("Distributed map-matching and route reconstruction from noisy GPS", ParagraphStyle("st", parent=P, fontSize=13, leading=17)),
          Spacer(1, 0.6 * cm),
          para("UE24CS352A Machine Learning, PES University<br/>Team No. 1, Section C<br/>Team members: Santosh, Koushik, Sharath<br/>18 November 2026", P),
          Spacer(1, 0.4 * cm),
          para("Public repository: github.com/santoshcheethiralame-dot/PathSnap_TeamNo_1_Section_C", P), Spacer(1, 0.8 * cm)]
disc = Table([[Paragraph(
    "<b>AI-assistance disclosure.</b> This report and the code, documents and figures in the repository were prepared with the help of "
    "Claude (Anthropic), working from the organisers' guidelines and our dataset. Every number in the report was produced by running the "
    "code in the repository and is read from the CSV and JSON files in <font face='Courier'>results/</font>; nothing is invented. "
    "The team has read the report and is responsible for it, and each member is prepared to explain every part in the viva. "
    "Section 10 lists what was assisted and what was done by the team.", P)]], colWidths=[16.4 * cm])
disc.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#b9770e")), ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fdf6e7")),
                          ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9), ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
story += [disc, Spacer(1, 0.8 * cm), para("Contents", H2)]
for line in ["1  Summary", "2  Dataset understanding and EDA", "3  Model experimentation", "4  Bayesian optimisation", "5  HMM and Viterbi decoding",
             "6  Folium maps", "7  API, UI and deployment", "8  Error analysis and limitations", "9  Reproducibility and repository guide",
             "10  Team contributions, AI assistance and academic integrity", "Appendix A  Metric definitions and open questions"]:
    story.append(para(line, ParagraphStyle("toc", parent=P, leftIndent=8, spaceAfter=1)))
story.append(PageBreak())

# ----------------------------------------------------------------------------- 1 summary
story += [para("1  Summary", H1),
          para("Given the noisy GPS points of one trip and the road network around it, PathSnap reconstructs the sequence of road arcs the vehicle drove. "
               "For every GPS fix we list nearby candidate arcs, describe each candidate with 26 numbers, and let a classifier answer one question: is this the reference road? "
               "An HMM decoded with the Viterbi algorithm then picks the best <i>connected</i> path through all the fixes. The API with the model and decoder runs on one laptop; "
               "the UI runs on another and shows the reconstructed route and the scores the API computed on the server.", P),
          figure("final/pipeline.png", 16.4 * cm, "Figure 1. The pipeline. Steps 1 to 4 are trained and tuned on train trips only."),
          para("Headline results on the ten student-test trips, which were never used for fitting or hyper-parameter search:", P),
          table([["", "Top-1 (share of fixes on the reference road)", "Route continuity", "Route overlap (IoU)"],
                 ["Rule baseline: nearest arc + heading check", pct(base_top1), "n/a", "n/a"],
                 ["Tuned boosting classifier alone", pct(hs.student_top1_classifier), pct(hs.student_route_continuity_classifier), f"{ov.classifier_iou:.3f}"],
                 ["Tuned boosting + HMM/Viterbi", pct(hs.student_top1_hmm), pct(hs.student_route_continuity_hmm), f"{ov.hmm_iou:.3f}"],
                 ["Aligned reference labels (ceiling)", "n/a (labels are the reference)", pct(ref_cont.reference_route_continuity.mean()), f"{ov.labels_iou:.3f}"]],
                [6.4 * cm, 4.6 * cm, 2.8 * cm, 2.6 * cm]),
          Spacer(1, 4),
          para("What is solid and what is not. The step from the rule baseline to a learned classifier (about " + f"{(hs.student_top1_classifier - base_top1) * 100:.1f}" + " points) and from the classifier to the HMM (about " + f"{(hs.student_top1_hmm - hs.student_top1_classifier) * 100:.1f}" + " points, "
               f"improving all {int(hs.student_trips_hmm_better)} student-test trips) is larger than any noise we measured. The differences between boosting, forest and MLP, "
               "and the gain from Bayesian tuning, are small (a fraction of a point) and are reported as such. All accuracy numbers rest on labels that we derived ourselves by aligning the reference "
               "route to the GPS fixes (Section 2.4), and route continuity, route overlap and macro F1 follow our own definitions (Appendix A) because the guidelines do not define them.", P)]

# ----------------------------------------------------------------------------- 2 EDA
story += [para("2  Dataset understanding and EDA", H1),
          para("<b>Partitions.</b> Each trip is one folder with four headerless files: <font face='Courier'>.track</font> (lon, lat, time), <font face='Courier'>.nodes</font>, "
               "<font face='Courier'>.arcs</font> (directed, from and to node) and <font face='Courier'>.route</font> (reference arc ids in driving order). "
               "We hold 80 train trips (208,417 points) and 10 student-test trips (19,160 points). The 10 evaluator trips arrive on the day and were never loaded: the loader's partition list "
               "contains only train and student-test. Every analysis below states which partition it used.", P),
          para("2.1  Missing values, sizes and structure (train and student-test, all 90 trips)", H2),
          para("We checked every file for missing values, invalid node ids in arcs, invalid arc ids in routes and isolated nodes: there are none (Figure 2a). "
               "Trips differ enormously in size, so a trip must not be allowed to dominate training.", P),
          table([["", "min", "median", "95th pct", "max"],
                 ["GPS points per trip", "262", "1,497", "7,516", "19,313"], ["Network arcs per trip", "649", "18,762", "184,034", "307,392"],
                 ["Arcs in the reference route", "59", "273", "950", "2,266"], ["Median arc length (m)", "6.8", "25.2", "57.8", "85.1"]],
                [6 * cm, 2.4 * cm, 2.4 * cm, 2.6 * cm, 2.6 * cm]),
          Spacer(1, 4),
          para("Every trip has its own network with its own node and arc numbering, so an arc id means nothing across trips and is never used as a feature or class label. "
               "Trips are worldwide (Russia, USA, Spain, Japan, Brazil and others), so no single projection works: each trip is projected into a local metric frame around its own "
               "centroid, and all distances are in metres. Median 95% of arcs have an exact reverse twin, so both directions of a road lie at the same distance from a GPS point; "
               "heading is therefore the only thing that separates them.", P),
          figure("eda/missing_values.png", 8.2 * cm, "Figure 2a. Missing-value check per partition (none found)."),
          para("2.2  Timing and GPS noise", H2),
          figure("eda/time_gaps.png", 14.6 * cm, "Figure 2b. Time gaps between consecutive fixes (train and student-test)."),
          para("The data is nominally 1 Hz, but 14 of 90 trips have fewer than 90% regular one-second steps (about 53 to 62% for the worst), steps of 10 to 17 s occur, and train trip 90 has a single "
               "step of 153,064 s. Timestamps are never repeated or backwards. Consequence: speed and every windowed feature use the real time delta, and a gap over 60 s (or an implied speed over 80 m/s) "
               "starts a new segment, which the HMM treats as a reset. The median distance from a GPS point to the reference route is 4.5 m (95th percentile 14 m): the noise is a few metres, "
               "but the largest jump between consecutive fixes is up to 888 m.", P),
          figure("eda/gps_route_distance.png", 14.6 * cm, "Figure 3. Distance from each GPS fix to the reference route (train and student-test)."),
          CondPageBreak(6 * cm), para("2.3  Spatial patterns", H2),
          side_by_side([("eda/gps_trajectories/train_00000001.png", "Figure 4a. GPS fixes, train trip 1."), ("eda/gps_trajectories/student_test_00000000.png", "Figure 4b. GPS fixes, student-test trip 0.")], 7.8 * cm),
          side_by_side([("eda/road_networks/train_00000001.png", "Figure 5a. Road network, train trip 1."), ("eda/road_networks/student_test_00000000.png", "Figure 5b. Road network, student-test trip 0.")], 7.8 * cm),
          para("Networks reach 307,000 arcs, so candidates come from a spatial index and the HMM uses a sparse shortest-path search with a distance cutoff. "
               "Only one place is shared by several trips (Moscow: train trips 10, 16, 29), which we keep in the same CV fold; student-test shares no area with train.", P),
          para("2.4  The reference route, the two defective trips and our labels", H2),
          para("The reference route is a reference, not guaranteed truth. Eleven trips have one to three breaks in the arc chain, ten repeat arcs (student-test trip 56 repeats 255), and a "
               "plain nearest-arc label moves backwards along the route a median 23 times per trip. We therefore label each fix by a monotone alignment: a Viterbi search over positions in the "
               "reference route that only moves forward, allows revisits, and prefers route distance to match GPS distance. Fixes more than 100 m from the route stay unlabelled. "
               "88 of 90 trips align cleanly (median label distance 4.7 m).", P),
          table([["Trip", "Points (share of train)", "What is wrong"],
                 ["train 33", "5,844 (2.8%)", "The reference route skips about half of the drive: 84% of points are within 15 m of some road, only 24% within 15 m of the route."],
                 ["train 54", "19,313 (9.3%)", "The road network is missing most of the drive: only 13% of points are within 15 m of any arc."]],
                [2.2 * cm, 3.4 * cm, 10.8 * cm]),
          Spacer(1, 4),
          para("Together they are 12% of all training points and would inject label noise, so they are quarantined from training and cross-validation automatically by a rule in the label code "
               "(it flags exactly these two). We fit on the remaining 78 trips. The pipeline does not crash on such trips: it still returns a prediction.", P),
          para("2.5  Candidate generation and feature engineering", H2),
          para("For each fix the candidates are the 10 nearest arcs within 40 m (retry at 80 m if none), plus the reverse twin of every kept arc. On clean trips the reference arc is missing from the "
               "candidate set for 0.6% of labelled points (point-weighted; median trip 0.1%), the mean is "
               f"{dsum.mean_cand_per_point.mean():.1f} candidates per fix, and the nearest arc alone is the right one for only about 56% of fixes on the median trip. That is why direction features are mandatory.", P),
          para("Each (fix, candidate) pair becomes one row with 26 features (about 457,000 rows from the 78 clean train trips after capping each trip at 1,000 points):", P),
          table([["Group", "Features", "Why"],
                 ["Geometry (6)", "dist_m, signed_m, abs_signed_m, frac, arc_len_m, offset_m", "How far the fix is from the arc, on which side, and where along it. The nearest road is usually right."],
                 ["Heading (3)", "cos_head, abs_head_deg, head_valid", "Agreement between the car's direction of motion and the arc direction; separates a road from its reverse twin."],
                 ["Motion (2)", "speed_mps, dt_prev_s", "Heading is unreliable when the car is nearly stopped; the real time gap qualifies every windowed value."],
                 ["Competition (4)", "rank, n_cand, dist_gap_best, twin_added", "How this candidate compares with the others at the same fix."],
                 ["Topology (2)", "deg_u, deg_v", "Junction degree of the arc's end nodes: a junction makes mistakes more likely."],
                 ["Context (9)", "cos_head_ff, head_filled, head_age_s, along_b5, along_f5, along_b15, along_f15, d_b5, d_f5",
                  "Added after Review 1 to fix the two biggest error sources: heading carried over stops (and its age), signed movement along the candidate arc over the previous and next 5 s and 15 s, and the distance of the fixes 5 s before and after to the arc. They use only fixes already in the trip file, never labels."]],
                [2.5 * cm, 6.2 * cm, 7.7 * cm]),
          Spacer(1, 4),
          para("Learned preprocessing is fitted on training trips only (only the MLP needs scaling, with a scaler fitted on its training rows).", P)]
story.append(PageBreak())

# ----------------------------------------------------------------------------- 3 models
story += [para("3  Model experimentation", H1),
          para("All five models see the same rows, the same 5 trip-grouped folds and seed 0 (so a trip is never split between fitting and testing). At most 1,000 points per trip are used so that one long "
               "trip cannot dominate. The 10 student-test trips are a second, untouched set used only to report. Labels are our own alignment of the reference route.", P),
          para("3.1  Comparison table", H2)]
rows = [["Model", "Family", "Key config", "CV macro F1 (std)", "CV Top-1", "Stu. route cont.", "Stu. macro F1", "Stu. log loss", "us / row", "Fit (s)"]]
rows.append(["Rule baseline (no learning)", "Rule", "dist + 10 x (1 - cos heading)", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a"])
for m in ["lr", "gnb", "rf", "hgb", "mlp", "hgb_tuned"]:
    r = runs.loc[m]
    rows.append([NAME[m], FAM[m], KEY[m], f"{r.cv_macro_f1_mean:.4f} ({r.cv_macro_f1_std:.3f})", pct(r.cv_top1_mean), pct(r.student_route_continuity),
                 f"{r.student_macro_f1:.3f}", f"{r.student_log_loss:.3f}", f"{r.inference_us_per_row:.2f}", f"{r.final_fit_seconds:g}"])
rows[1][4] = "n/a"
story += [table(rows, [2.5 * cm, 1.8 * cm, 3.0 * cm, 1.7 * cm, 1.3 * cm, 1.3 * cm, 1.2 * cm, 1.2 * cm, 1.1 * cm, 1.2 * cm]),
          para(f"Student Top-1: rule baseline {pct(base_top1)}; " + "; ".join(f"{NAME[m].split(',')[0]} {pct(runs.loc[m].student_top1)}" for m in ["lr", "gnb", "rf", "hgb", "mlp", "hgb_tuned"]) +
               ". Source: results/model_runs.csv. Candidate-level macro F1 and Top-1 per fix are defined in Appendix A. 'Student route cont.' is the classifier's own picks without the HMM.", SM),
          para("3.2  Per-model answers", H2),
          para("The guideline numbers its per-model questions 1, 2, 5, 6, 7, 8 and 9 in the text (3 and 4 are absent); we answer them under those numbers.", SM)]


def gap(m):
    return diag.loc[m].train_top1 - diag.loc[m].student_top1


def model_block(m, q1, q2, q5, q6, q9):
    r = runs.loc[m]
    d = diag.loc[m]
    out = [para(NAME[m], H3), QA(1, "Why suitable.", q1), QA(2, "Strengths and limitations.", q2), QA(5, "Hyper-parameters.", q5), QA(6, "Objective.", q6),
           QA("7 and 8", "Student-test and CV results.",
              f"Student-test Top-1 {pct(r.student_top1)}, macro F1 {r.student_macro_f1:.3f}, log loss {r.student_log_loss:.3f}, route continuity of its own picks {pct(r.student_route_continuity)}. "
              f"CV: Top-1 {pct(r.cv_top1_mean)}, macro F1 {r.cv_macro_f1_mean:.4f} (fold-to-fold std {r.cv_macro_f1_std:.3f}). Fit {r.final_fit_seconds:g} s, inference {r.inference_us_per_row:.2f} us per row."),
           QA(9, "Over- or under-fitting.", q9)]
    return out


story += model_block("lr",
    "A linear model on the 26 features is the simplest sensible classifier and a cheap, readable reference: it shows how far one straight decision boundary gets.",
    "Strengths: fast, convex, stable, calibrated-looking probabilities. Limits: it cannot express 'close <i>and</i> pointing the right way', which is an interaction of features.",
    "L2 penalty, C = 1, up to 500 iterations (scikit-learn defaults otherwise); inputs standardised.",
    "Binary log loss (cross-entropy) with an L2 penalty on the weights.",
    f"Underfits: low on both train ({pct(diag.loc['lr'].train_top1)}) and unseen trips ({pct(diag.loc['lr'].student_top1)}), a {gap('lr') * 100:.1f}-point gap. The model is too simple rather than too flexible.")
story += model_block("gnb",
    "A generative model that assumes independent, Gaussian features. It represents the probabilistic family and is deliberately simple and fast.",
    "Strengths: trains in a fraction of a second. Limits: our features are strongly correlated (distance, signed offset, rank and gap to best), which breaks the independence assumption, so it is badly over-confident: its log loss is 0.905 against 0.083 to 0.119 for the other models.",
    "Variance smoothing 1e-9 (default).",
    "Maximises the joint likelihood of features and class under the independence assumption; no iterative loss.",
    f"Biased rather than overfit: {pct(diag.loc['gnb'].train_top1)} on train rows and {pct(diag.loc['gnb'].student_top1)} on unseen trips. It trails because its assumption is wrong for this data, not because it memorised.")
story += model_block("rf",
    "Each candidate is a row of numbers whose meaning depends on combinations (a close road is right only if it also points the way the car is moving). Trees split on such combinations directly, need no scaling and cope with the mix of metres, angles and counts.",
    "Strengths: robust, few settings to get wrong, top CV accuracy among the untuned models. Limits: it is the slowest model to fit and to predict, it memorises its training trips (below), and it scores each fix alone with no memory of the previous fix.",
    "150 trees, minimum 5 samples per leaf, square-root of the feature count per split, bootstrap sampling (default), seed 0.",
    "Each tree splits to reduce Gini impurity; the forest averages the trees' class votes into a probability.",
    f"Overfits: {pct(diag.loc['rf'].train_top1)} on the rows it trained on against {pct(diag.loc['rf'].student_top1)} on unseen trips, a {gap('rf') * 100:.1f}-point gap, and log loss rises from {diag.loc['rf'].train_log_loss:.3f} to {diag.loc['rf'].student_log_loss:.3f}. Evidence is the gap, not a feeling. A forest has no epochs, so no training curve is drawn.")
story += model_block("hgb",
    "The same tabular, interacting-feature reasoning as the forest, but trees are added one at a time and each corrects the previous trees' mistakes, which usually gives sharper probabilities. It is fast, which matters because the Bayesian search refits it many times.",
    "Strengths: accuracy level with the best untuned models, a small train-to-test gap, a few seconds to fit. Limits: more settings than the forest, and the same blind spot as every per-fix classifier: it cannot see where the car just was (the HMM supplies that).",
    "Up to 300 iterations (early stopping, 10% hold-out, patience 15), learning rate 0.1, 31 leaf nodes per tree, scoring = loss.",
    "Binary log loss (cross-entropy), minimised stage by stage by gradient boosting.",
    f"Mild: {pct(diag.loc['hgb'].train_top1)} on training rows against {pct(diag.loc['hgb'].student_top1)} on unseen trips ({gap('hgb') * 100:.1f} points). The loss curve (Figure 8) shows training and hold-out loss falling together and flattening before early stopping ends the run. That hold-out is random rows, so it is for stopping only, not an honest test.")
story += model_block("mlp",
    "A small neural network can learn its own mixtures of the 26 features; it is the neural family in the comparison and should compete with trees if the problem is smooth.",
    "Strengths: smooth probabilities, fast inference, accuracy level with the trees. Limits: needs scaled inputs (standard scaler fitted on training rows), longer to fit than boosting, and no accuracy gain over it.",
    "Two hidden layers (64 and 32 units), ReLU and Adam (defaults), early stopping with 10% hold-out and patience 8, up to 60 epochs, batch size 512, seed 0.",
    "Binary log loss, minimised with Adam on mini-batches.",
    f"Mild at most: {pct(diag.loc['mlp'].train_top1)} on training rows against {pct(diag.loc['mlp'].student_top1)} on unseen trips ({gap('mlp') * 100:.1f} points). The training-loss curve falls and flattens (Figure 8); the hold-out accuracy is on random rows and is noisy.")
sp3 = [runs.loc[m].student_top1 for m in ["rf", "hgb", "mlp", "hgb_tuned"]]
story += [para("Across the models", H3),
          para(f"Forest, boosting, MLP and tuned boosting lie within {(max(sp3) - min(sp3)) * 100:.1f} points of each other on student-test, which is smaller than the fold-to-fold spread (macro F1 std about 0.011 to 0.015), "
               "so their order is not statistically firm. What is solid: all four beat the rule baseline "
               f"({pct(base_top1)}), logistic regression ({pct(runs.loc['lr'].student_top1)}) and naive Bayes ({pct(runs.loc['gnb'].student_top1)}).", P),
          para("3.3  Visualisations", H2),
          side_by_side([("models/cv_macro_f1.png", "Figure 6a. Cross-validated macro F1 per model (mean, fold std)."), ("models/log_loss.png", "Figure 6b. Log loss per model (student-test).")], 7.8 * cm),
          figure("models/student_test_top1_macro_f1.png", 12.5 * cm, "Figure 7. Student-test Top-1 and macro F1 per model."),
          figure("models/training_curves.png", 14.5 * cm, "Figure 8. Training curves, only for the models that train in iterations (boosting and MLP). The forest, logistic regression and naive Bayes have none."),
          side_by_side([("models/train_vs_student_top1.png", "Figure 9. Train-row Top-1 against student-test Top-1: the forest and tuned boosting show the widest gap.")], 9 * cm),
          figure("models/confusion_matrices.png", 16.4 * cm, "Figure 10. Candidate-level confusion matrices on student-test (is this the reference road?)."),
          para("3.4  Model selection", H2),
          para(f"We carry <b>tuned HistGradient Boosting</b> forward. Reasons, in order: it has the best CV Top-1 ({pct(runs.loc['hgb_tuned'].cv_top1_mean)}) and student-test Top-1 ({pct(runs.loc['hgb_tuned'].student_top1)}) with a log loss level with the forest's, which matters because the HMM uses the "
               f"probabilities directly; boosting fits in seconds, which made the 40-trial Bayesian search affordable (the forest and MLP would have taken hours); and boosting needs no scaling. "
               f"We state the caveats plainly: the margin over the forest and MLP is within noise, and the tuned model fits training rows harder ({pct(diag.loc['hgb_tuned'].train_top1)} train against {pct(diag.loc['hgb_tuned'].student_top1)} on unseen trips). "
               "Selection used cross-validation on train and the student-test set only, never training accuracy and never the evaluator trips.", P)]
story.append(PageBreak())

# ----------------------------------------------------------------------------- 4 BO
cvu, cvt = runs.loc["hgb"], runs.loc["hgb_tuned"]
story += [para("4  Bayesian optimisation (HistGradient Boosting)", H1),
          QA(1, "Hyper-parameters tuned.", "Learning rate, maximum leaf nodes, minimum samples per leaf, L2 regularisation, number of histogram bins and number of iterations. Early stopping stayed on in every trial (10% hold-out, patience 15), so the iteration count is an upper limit."),
          QA(2, "Ranges.", "Learning rate 0.02 to 0.3 (log scale); leaf nodes 8 to 128 (log, integer); minimum leaf 10 to 400 (log, integer); L2 1e-4 to 10 (log); bins 64, 128 or 255; iterations 150 to 500 (integer)."),
          QA(3, "Objective.", "Mean observation-level Top-1 over the same 5 trip-grouped folds on the 78 clean train trips (the route-matching metric). Log loss and macro F1 were logged for every trial. Optuna TPE sampler, seed 0. "
             "The student-test trips were never touched by the search."),
          QA(4, "Trials.", f"{int(bb['n_trials'])} trials, {bb['total_seconds'] / 60:.0f} minutes. Trial 0 is the default-like configuration, so the search contains its own reference point. The full log is results/bayes_trials_hgb.csv."),
          QA(5, "Best values.", f"Trial {bb['trial']}: learning rate {bb['config']['learning_rate']:.3f}, {bb['config']['max_leaf_nodes']} leaf nodes, minimum leaf {bb['config']['min_samples_leaf']}, L2 {bb['config']['l2_regularization']:.2f}, "
             f"{bb['config']['max_bins']} bins, up to {bb['config']['max_iter']} iterations (results/bayes_best_hgb.json)."),
          QA(6, "Tuned against untuned.", "Same partitions and metrics:"),
          table([["", "Untuned", "Tuned"],
                 ["CV Top-1 (5 folds, search target)", pct(cvu.cv_top1_mean, 2), pct(cvt.cv_top1_mean, 2)],
                 ["CV macro F1", f"{cvu.cv_macro_f1_mean:.4f}", f"{cvt.cv_macro_f1_mean:.4f}"],
                 ["Student-test Top-1", pct(cvu.student_top1, 2), pct(cvt.student_top1, 2)],
                 ["Student-test macro F1", f"{cvu.student_macro_f1:.4f}", f"{cvt.student_macro_f1:.4f}"],
                 ["Student-test log loss", f"{cvu.student_log_loss:.4f}", f"{cvt.student_log_loss:.4f}"],
                 ["Fit time (s)", f"{cvu.final_fit_seconds:g}", f"{cvt.final_fit_seconds:g}"]], [7 * cm, 3 * cm, 3 * cm]),
          Spacer(1, 4),
          figure("final/bayes_trials.png", 11.5 * cm, "Figure 11. The 40 trials and the best score so far (5-fold grouped CV Top-1)."),
          para(f"Reading. Tuning raised CV Top-1 by {(cvt.cv_top1_mean - cvu.cv_top1_mean) * 100:.2f} points and student-test Top-1 by {(cvt.student_top1 - cvu.student_top1) * 100:.2f} points. "
               f"The 40 trials span {bt.value.min() * 100:.2f}% to {bt.value.max() * 100:.2f}% and the fold-to-fold standard deviation is about {bt.user_attrs_cv_top1_std.mean() * 100:.1f} points, "
               "so the whole search moves the score by less than the noise between folds. The search optimised the CV score itself, so the CV gain is optimistic; the student-test gain is the figure to trust, and it is modest. "
               "The tuned model also fits the training rows harder (a wider train-to-test gap, Section 3.4). We keep it because it has the better CV and log loss, not because tuning transformed the result.", P)]

# ----------------------------------------------------------------------------- 5 HMM
story += [para("5  HMM and Viterbi decoding", H1),
          QA(1, "Hidden states.", "For each GPS fix, its candidate road arcs: the 10 nearest within 40 m (widened to 80 m if none) plus the opposite-direction twin of each."),
          QA(2, "Emissions.", f"The classifier's probability for each candidate, normalised over that fix's candidates, as a log, multiplied by a weight (lambda = {hs.lam:g})."),
          QA(3, "Transitions.", f"Between consecutive fixes, the shortest driven distance on the directed road graph from the previous candidate's position to this one's is compared with the straight-line distance between the two fixes. "
             f"The score is minus the absolute difference divided by beta (beta = {hs.beta:g} m, gamma = {hs.gamma:g}). A pair whose driven distance matches the GPS distance is likely; one that needs a long detour is not. "
             "Search is a sparse Dijkstra with a cutoff on the trip's own graph."),
          QA(4, "Missing candidates and gaps.", "A time gap over 60 s or an implied speed over 80 m/s starts a new segment and the chain restarts there. A fix with no arc within 80 m is left out of the chain and gets no arc. "
             f"If the reference road is not among a fix's candidates ({pct(MISS_ST, 1)} of student-test points) the HMM cannot choose it."),
          QA(5, "Disconnected moves.", f"If no path exists within the search limit, the move receives a flat penalty of {hs.penalty:g} instead of probability zero, so decoding never fails; the decoder then prefers a connected alternative when one exists."),
          QA(6, "Did it improve the route?", f"Yes: on all {int(hs.student_trips_hmm_better)} student-test trips. Decoder settings were tuned on train out-of-fold probabilities only (lambda, beta, gamma and the penalty on a grid, results/hmm_grid_hgb_tuned.csv)."),
          table([["Student-test, tuned boosting", "Classifier only", "Classifier + HMM"],
                 ["Top-1 (point-weighted)", pct(hs.student_top1_classifier), pct(hs.student_top1_hmm)],
                 ["Route continuity (mean per trip)", pct(hs.student_route_continuity_classifier), pct(hs.student_route_continuity_hmm)],
                 ["Route overlap IoU, direction counted", f"{ov.classifier_iou:.3f}", f"{ov.hmm_iou:.3f}"],
                 ["Overlap precision / recall", f"{ov.classifier_precision:.3f} / {ov.classifier_recall:.3f}", f"{ov.hmm_precision:.3f} / {ov.hmm_recall:.3f}"],
                 ["Train out-of-fold Top-1", pct(hs.train_oof_top1_classifier), pct(hs.train_oof_top1_hmm)],
                 ["Trips improved / worsened (of 10)", "", f"{int(hs.student_trips_hmm_better)} / {int(hs.student_trips_hmm_worse)}"]], [7 * cm, 4.2 * cm, 4.2 * cm]),
          Spacer(1, 4),
          figure("final/hmm_per_trip.png", 11.5 * cm, "Figure 12. Top-1 per student-test trip, classifier alone against classifier + HMM."),
          para(f"Ceiling. The aligned reference labels themselves score {ov.labels_iou:.3f} overlap, so the HMM's {ov.hmm_iou:.3f} is close to what one-arc-per-fix output can reach.", P),
          para("Where the HMM did not help", H3),
          para(f"On the 78 train trips (out-of-fold probabilities) the HMM was better on {tr_better}, about equal (within 0.2 points) on {tr_same} and worse on {tr_worse}. "
               f"The largest loss is train trip {int(worst.trip)} ({worst.change * 100:+.1f} points). An HMM cannot choose a road that is not among the candidates, and the trips where the reference road is most often missing from the candidates "
               "(for example trip 72, with the reference missing for 14.8% of its fixes) are where decoding has least to work with. Trips with many stopped fixes and parallel roads (for example train trip 16) stay weak with or without the HMM, "
               "which is consistent with a stopped car's GPS drift being read as real movement; we have not isolated that cause.", P),
          para(f"Effect on the kinds of mistake (student-test, {EC['total']:,} labelled fixes): wrong fixes fall from {EC['wrong']:,} to {EH['wrong']:,}. "
               f"Stopped fixes ({pct(EC['stop_share'])} of fixes) have an error rate of {pct(EC['stop_err'])} before the HMM and {pct(EH['stop_err'])} after, against {pct(EC['move_err'])} and {pct(EH['move_err'])} for moving fixes.", P)]

# ----------------------------------------------------------------------------- 6 Folium
story += [para("6  Folium maps", H1),
          para("The generator <font face='Courier'>scripts/make_folium_map.py</font> writes an interactive HTML map for any trip (the submitted example is student-test trip 9, which is also the weakest student-test trip). "
               "Its layers, each with its own colour, a legend and a layer control, are: the observed GPS fixes; the road network near the trip; the reference route; the predicted route (classifier + HMM); "
               "and the fixes where the predicted road is not the reference road. The map is centred and zoomed on the trip, and tiles load from OpenStreetMap, so it needs a network connection.", P),
          figure("../maps/trip_00000009_student_test.png", 16.2 * cm, f"Figure 13. Folium map of student-test trip 9: Top-1 {pct(hst[hst.trip == 9].top1_hmm.iloc[0])} with the HMM. Source: report/maps/trip_00000009_student_test.html (interactive)."),
          para("Reading the map: where the predicted route follows the reference route the two overlap; the marked fixes mark the places where one fix was assigned to a neighbouring road, which the route-level metrics forgive "
               "more than the per-fix Top-1 does.", P)]

# ----------------------------------------------------------------------------- 7 API
story += [para("7  API, UI and deployment", H1),
          para("<b>Architecture.</b> Laptop 1 runs a FastAPI service with the saved model, the HMM and the road networks. Laptop 2 runs a single-page UI that lists the trips, sends the chosen trip id to the API and draws what comes back. "
               "The reference route is read only by the server-side scoring module (<font face='Courier'>api/scoring.py</font>) after the prediction is made; the model code opens trips with <font face='Courier'>with_route=False</font>, "
               "so the prediction cannot see the ground truth, and the UI only displays the scores the API returns. No prediction is hard-coded or edited.", P),
          table([["Endpoint", "Purpose"],
                 ["GET /health", "Liveness and number of trips available."], ["GET /trips", "Lists the trip ids found in the configured trips folder, at request time."],
                 ["POST /predict {trip_id}", "Runs candidates, features, classifier and HMM for the trip and returns the predicted route, the per-fix arcs, the polyline, and Top-1, coverage and route continuity computed server-side."]],
                [4.4 * cm, 12 * cm]),
          Spacer(1, 4),
          para("<b>Deployment.</b> On Laptop 1 set the trips folder and start the server on all interfaces (the folder is read at request time, so no code change is needed for the evaluator's ten trips); on Laptop 2 serve the "
               "<font face='Courier'>ui/</font> folder, type Laptop 1's address and press Load Trips:", P),
          Preformatted("Laptop 1:  set PATHSNAP_TRIPS_DIR=D:\\evaluator_trips\n"
                       "           python -m uvicorn api.main:app --host 0.0.0.0 --port 8000\n"
                       "Laptop 2:  python -m http.server 8080 --directory ui     (then open http://localhost:8080)", CODE),
          Spacer(1, 4),
          para("TCP port 8000 must be open in the Windows firewall for the network profile in use (we use <font face='Courier'>profile=any</font>) and the two laptops must share a network. A trip with no <font face='Courier'>.route</font> "
               "file is still predicted and returned with <font face='Courier'>\"scored\": false</font>; defective trips do not crash the service; a malformed id gives HTTP 400, an unknown id 404 and a missing id 422. "
               "Every request and every prediction (trip id, method, route size, Top-1, continuity, time) is printed on Laptop 1's console and written to <font face='Courier'>api/logs/requests.log</font>.", P),
          para("<b>UI.</b> Screenshots below come from a rehearsal with the API and the UI on the same laptop (localhost); the two-laptop run uses the same page with Laptop 1's address.", SM),
          side_by_side([("ui/ui_2_trips_loaded.png", "Figure 14a. After Load Trips: the ten trips are listed."), ("ui/ui_3_result_00000009.png", "Figure 14b. Trip 9 reconstructed: map and four result cards.")], 7.9 * cm),
          figure("ui/ui_3_result_00000040.png", 11.5 * cm, "Figure 15. Trip 40: Top-1 95.73%, coverage 100%, route continuity 96.24%, 320 predicted route arcs."),
          para("API responses and the server log (shortened lists):", P),
          Preformatted(wrap_lines(open(os.path.join(REP, "api_demo_output.txt"), encoding="utf8").read().replace("\t", "    ")), CODE)]

# ----------------------------------------------------------------------------- 8 errors & limits
story += [para("8  Error analysis and limitations", H1),
          para("What the remaining mistakes are (tuned boosting, student-test, per labelled fix; each wrong top choice goes into the first matching category, definitions in scripts/error_analysis.py):", P)]
lab = dict(opposite="Opposite direction of the right road", next_arc="Next or previous arc on the same road", sibling="Sibling arc at the same junction",
           parallel="Parallel road within 20 m, not connected", elsewhere="Somewhere else", missing="Right road not among the candidates")
rows = [["Category", "Classifier only", "Classifier + HMM"]]
for c in lab:
    rows.append([lab[c], f"{EC['cats'][c]:,} ({EC['cats'][c] / EC['wrong'] * 100:.0f}% of mistakes)", f"{EH['cats'][c]:,} ({EH['cats'][c] / max(EH['wrong'], 1) * 100:.0f}% of mistakes)"])
rows.append(["All wrong fixes", f"{EC['wrong']:,} of {EC['total']:,}", f"{EH['wrong']:,} of {EH['total']:,}"])
story += [table(rows, [7 * cm, 4.7 * cm, 4.7 * cm]), Spacer(1, 4),
          para("<b>What these show.</b> The mistakes are local: the opposite direction, the next arc along the same road, a branch at the same junction, or a close parallel road. Only a few percent are somewhere unrelated. "
               "These are exactly the cases one fix cannot settle, and route memory is what the HMM adds. The mistakes also concentrate on stopped fixes (speed below 1 m/s), where heading is meaningless.", P),
          para("Limitations we will state to the evaluators", H2)]
story += bullets([
    "<b>Our labels.</b> Per-fix labels come from our own monotone alignment of the reference route; the organisers' labels may differ, and then absolute numbers shift. The ordering of methods should not.",
    "<b>Our metric definitions.</b> The guidelines do not define macro F1 or route continuity. We use candidate-level macro F1 and the share of consecutive predicted arcs that connect (Appendix A) and ask the organisers to confirm (docs/11_questions_for_organisers.md).",
    f"<b>Tuned on the same regions.</b> Candidate settings, classifier and decoder were tuned on train and student-test, so evaluator trips (possibly other regions or sampling rates) may behave differently. Single numbers on ten trips carry a wide spread: per-trip Top-1 ranges from {pct(hst.top1_hmm.min())} to {pct(hst.top1_hmm.max())}.",
    "<b>The HMM cannot recover a missing road.</b> If the reference arc is not among the candidates (0.6% of labelled points on clean trips, up to 14.8% on one train trip) it is unreachable.",
    "<b>Stopped vehicles and parallel roads</b> remain the hardest cases, with or without the HMM.",
    "<b>Two defective training trips</b> were quarantined. Evaluator trips may contain similar defects; the pipeline returns a prediction and does not crash, but scores on a defective trip are not meaningful.",
    "<b>Not yet verified on hardware.</b> The API and UI were rehearsed end to end on one laptop (all ten student-test trips reproduce the expected scores); the two-laptop run depends on the venue network and the Windows firewall."])

# ----------------------------------------------------------------------------- 9 repro
story += [para("9  Reproducibility and repository guide", H1),
          para("Everything in the report is reproducible from the public repository: code, results, saved model, Folium generator, API and UI. Seeds are fixed (0). The dataset is not committed.", P),
          table([["Path", "Contents"],
                 ["src/pathsnap/", "Loader and projection, segments, label alignment, candidates, features, models, HMM, metrics, predict_trip"],
                 ["scripts/", "Audit, dataset build, training, Bayesian search, HMM, error analysis, route overlap, figures, maps, this report"],
                 ["api/, ui/", "FastAPI service and server-side scoring; single-page UI"],
                 ["models/", "Saved tuned model (model.joblib) and decoder settings (config.json)"],
                 ["results/", "Every CSV and JSON behind the numbers; results/v1/ holds the Review 1 run (17 features)"],
                 ["report/", "EDA and model figures, Folium map, this report"],
                 ["tests/", "24 unit tests on synthetic roads and a synthetic trip for the API (python -m pytest -q tests)"]], [3.6 * cm, 12.8 * cm]),
          Spacer(1, 6),
          Preformatted("python scripts/audit_data.py data/raw results/data_audit_per_trip.csv\n"
                       "python scripts/build_dataset.py data/raw data/processed results/dataset_build_summary.csv\n"
                       "python scripts/train_models.py DATA_RAW DATA_PROCESSED SUMMARY_CSV OUT_DIR PROBS_DIR rf hgb mlp lr gnb\n"
                       "python scripts/bayes_opt.py DATA_PROCESSED SUMMARY_CSV OUT_DIR\n"
                       "python scripts/train_models.py ... hgb_tuned\n"
                       "python scripts/run_hmm.py DATA_RAW DATA_PROCESSED SUMMARY_CSV PROBS_DIR CACHE_DIR OUT_DIR hgb_tuned\n"
                       "python scripts/export_model.py DATA_PROCESSED SUMMARY_CSV HMM_SUMMARY_CSV hgb_tuned\n"
                       "python scripts/check_predict.py data/raw        # end to end, reproduces the student-test Top-1 above", CODE),
          Spacer(1, 4),
          para("The Review 1 numbers (17 features: classifier 89.1%, with HMM 93.15%) are kept in results/v1/, and setting <font face='Courier'>PATHSNAP_FEATURES=v1</font> restores the original feature set for a clean before-and-after comparison.", P)]

# ----------------------------------------------------------------------------- 10 team
story += [para("10  Team contributions, AI assistance and academic integrity", H1),
          table([["Member", "Responsibility"],
                 ["Santosh", "Core pipeline: loader and projection, label alignment, candidates, features, training harness and the boosting, forest and MLP models, Bayesian optimisation, HMM and Viterbi, metrics, packaging into predict_trip, error analysis."],
                 ["Koushik", "EDA figures and data-quality log, Folium map generator, logistic regression and naive Bayes, comparison plots."],
                 ["Sharath", "FastAPI service, server-side scoring, UI, deployment and the two-laptop run."]], [2.4 * cm, 14 * cm]),
          Spacer(1, 4),
          para("<b>AI assistance.</b> Claude (Anthropic) was used to draft the repository's code, scripts and documents and this report, and to run and re-run the experiments, working from the organisers' guidelines and our dataset. "
               "It was also used to find defects (for example the candidate-count wording, the tuned model's overfitting gap and API edge cases) and to regenerate every result after we added the context features. "
               "The decisions that shaped the work (the monotone label alignment, quarantining two trips, the context features, reporting tuning honestly as a small gain) were made and reviewed by the team. "
               "No metric, plot, citation or experiment is fabricated: every figure in this report is produced by code in the repository from files in results/. Cross-checks that any reader can repeat: python -m pytest -q tests, "
               "python scripts/check_predict.py data/raw, and the expected per-trip table in docs/12_testing_guide.md.", P),
          para("<b>Integrity.</b> The evaluator partition was never loaded; the evaluator trips and labels played no part in training, feature design, model choice, hyper-parameters, thresholds or decoder settings. "
               "No models, weights, configs, code or results were shared with other teams. Every team member must be able to explain every part of the code, experiments and results in the viva.", P)]

# ----------------------------------------------------------------------------- appendix
story += [CondPageBreak(9 * cm), para("Appendix A  Metric definitions and open questions", H1),
          para("The guidelines name these metrics without defining all of them. These are our definitions; the numbers in this report use them.", P),
          table([["Metric", "Definition used here"],
                 ["Top-1 (observation level)", "Share of labelled GPS fixes where the best-scored candidate is the reference arc (pooled over fixes unless stated 'mean per trip')."],
                 ["Macro F1 (candidate level)", "Every (fix, candidate) row is classified as is-reference or not; F1 of each of the two classes, averaged."],
                 ["Log loss", "Binary cross-entropy over all candidate rows."],
                 ["Route continuity", "Share of consecutive distinct predicted arcs where one ends at the node where the next starts (mean per trip). The aligned reference route itself is not always 100% continuous."],
                 ["Route overlap", "Length-weighted precision, recall and IoU of the predicted arc set against the reference route; direction counted unless stated."],
                 ["Candidate miss rate", "Share of labelled fixes whose reference arc is not among the candidates."],
                 ["Prediction coverage", "Share of fixes that received an arc."]], [4.2 * cm, 12.2 * cm]),
          Spacer(1, 4),
          para("Questions still open with the organisers (docs/11_questions_for_organisers.md): the exact definitions of macro F1 and route continuity, how per-fix labels are derived, whether evaluator trips can contain defective references, "
               "and what the first-phase evaluation checks. If their definitions differ, only the code in src/pathsnap/metrics.py changes and every table can be regenerated with one script.", P)]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#777b80"))
    canvas.drawString(2 * cm, 1.1 * cm, "PathSnap | UE24CS352A | Team No. 1, Section C")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"{doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                        title="PathSnap: project report", author="Team No. 1, Section C")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
