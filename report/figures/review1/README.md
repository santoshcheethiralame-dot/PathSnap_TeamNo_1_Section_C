# Review 1 figures

Made by `scripts/make_review1_figures.py` from `results/` and the raw train/student-test trips. Colours follow one fixed slot per model.

| Slide | Figures |
|---|---|
| 1 Problem | 01_trip_map, 02_trips_world |
| 2 Data | 04_sampling_gaps, 05_gps_noise, 06_defective_trips |
| 3 Framing | 07_candidates_example, 08_pipeline |
| 4 Evaluation | 09_partitions_and_cv, 10_points_per_trip |
| 5 Models | 17_training_curves |
| 6 Accuracy | 03_accuracy_ladder, 11_cv_by_fold, 12_student_test_by_trip, 13_time_vs_accuracy |
| 7 What the numbers mean | 14_overfitting, 15_calibration, 16_error_breakdown, 18_confusion_matrices |

`report/PathSnap_Review1_handout.pdf` is a 3-page summary built by `scripts/make_review1_report.py`.

Note: these figures and the handout use the Review 1 run (17 features). Their source CSVs are in `results/v1/`; `results/` holds the final run.
