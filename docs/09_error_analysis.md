# 09 · Error analysis and route overlap

Reproduce: `python scripts/error_analysis.py ...` and `python scripts/route_overlap.py ...` (arguments in each file's docstring). Outputs: `results/error_analysis_per_trip.csv`, `results/route_overlap_student_test.csv`. Tuned boosting, labels are our own alignment.

## What the "different segment" mistakes are
Each wrong top choice goes into the first matching category (definitions in `scripts/error_analysis.py`).

| Category | Student-test | Train out-of-fold |
|---|---|---|
| Opposite direction of the right road | 32.1% | 29.9% |
| Next or previous arc on the same road | 26.5% | 28.1% |
| Sibling arc at the same junction | 18.8% | 14.1% |
| Parallel road within 20 m, not connected | 17.4% | 18.6% |
| Somewhere else | 2.8% | 3.9% |
| Right road not among the candidates | 2.3% | 5.4% |
| Wrong fixes in total | 2,090 of 19,157 (10.9%) | 19,256 of 182,231 (10.6%) |

So the earlier "68% a different segment" splits into three small, local kinds: one arc along the same road (about 27%), a branch at the same junction (about 17%) and a close parallel road (about 18%). Only about 3% are somewhere unrelated. These are the cases a single fix cannot settle, because both arcs are nearby and point nearly the same way. Route memory is what the HMM adds.

(The Review 1 chart `16_error_breakdown.png` uses the untuned model: 650 opposite, 1,382 other, 49 missing. The table above uses the tuned model, so counts differ slightly.)

## Stopped fixes
Fixes with speed below 1 m/s are 7.7% of student-test fixes (8.2% of train) but 29% (34%) of the mistakes. Derived from those shares, the error rate is about 42% for stopped fixes against about 8% for moving ones. Heading is unreliable when the car is not moving, so the direction feature loses its power. Not yet checked: how much of this the HMM repairs.

## Trip 16 (train, 53% Top-1)
- 1,305 labelled fixes, one segment, median distance to the reference road 6.7 m, so GPS noise is ordinary.
- 20% of its fixes are stopped (train average 8%).
- Its mistakes: 230 parallel road, 196 opposite direction, 58 next arc, 46 sibling, 72 elsewhere, 13 missing.
- Across the 78 train trips, Top-1 correlates with the share of stopped fixes (Spearman -0.71) and with candidates per fix (-0.56, denser roads), and hardly with noise distance (-0.27) or trip length (0.02).
- The other weak trips fit: trip 10 has 35% stopped fixes (Top-1 62%), trip 61 has 23% (67%), trip 72 has 24% (69%) and also 120 missing candidates.

So trip 16 is mostly a stopped-vehicle and parallel-road case, not a data defect. This is correlation over 78 trips, not a proof.

## Route overlap (student-test, length-weighted against the reference route)
| | Classifier only | Classifier + HMM | Aligned labels (ceiling) |
|---|---|---|---|
| IoU, direction counted | 0.892 | 0.955 | 0.986 |
| IoU, direction ignored | 0.913 | 0.966 | 0.986 |
| Precision (share of predicted length on the reference) | 0.909 | 0.973 | 1.000 |
| Recall (share of reference length predicted) | 0.979 | 0.980 | 0.986 |

The HMM raises precision (fewer wrong roads in the route) and leaves recall unchanged. The ceiling is below 1 because sampling skips some short arcs. Overlap is our own definition (see `05_open_questions.md`).
