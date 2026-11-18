"""Train and evaluate models from the registry.

    python scripts/train_models.py DATA_RAW DATA_PROCESSED SUMMARY_CSV OUT_DIR PROBS_DIR rf hgb mlp

Trip-grouped 5-fold CV on non-quarantined train trips, a final fit on all of
them, then student-test evaluation. Results go to OUT_DIR (model_runs.csv,
cv_folds.csv, student_test_per_trip.csv, history_<model>.json). Out-of-fold
probabilities for the HMM go to PROBS_DIR (not committed).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from pathsnap.modeling import run_model  # noqa: E402


def main():
    raw, proc, summary, out, probs, *names = sys.argv[1:]
    for name in names:
        row = run_model(name, proc, summary, raw, out, probs, log=lambda *a: print(*a, flush=True))
        print("DONE", name, {k: (round(v, 4) if isinstance(v, float) else v)
                             for k, v in row.items() if k not in ("config", "family")}, flush=True)


if __name__ == "__main__":
    main()
