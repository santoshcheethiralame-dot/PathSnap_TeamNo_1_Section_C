"""Model registry. Each entry: family, a config dict (logged with every run),
a build(config) function returning an sklearn estimator, and an optional
history(estimator) returning epoch-wise curves for models that have them.

To add a model, copy one entry, change family/config/build, and run
    python scripts/train_models.py <name> ...
"""
import json
import os

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0


def _rf(c):
    return RandomForestClassifier(**c, n_jobs=-1, random_state=SEED)


def _hgb(c):
    return HistGradientBoostingClassifier(**c, random_state=SEED)


def _mlp(c):
    return make_pipeline(StandardScaler(), MLPClassifier(**c, random_state=SEED))


def _hgb_history(est):
    # scoring="loss": the scores are negative log loss, on the training rows and
    # on the internal early-stopping hold-out (random rows, not whole trips)
    return dict(train_score=list(map(float, est.train_score_)), val_score=list(map(float, est.validation_score_)),
                note="negative log loss per boosting iteration")


def _mlp_history(est):
    m = est[-1]
    return dict(train_loss=list(map(float, m.loss_curve_)), val_score=list(map(float, m.validation_scores_)),
                note="train_loss is log loss per epoch; val_score is accuracy on a random-row hold-out")


REGISTRY = {
    "rf": dict(
        family="Ensemble (bagging of trees)",
        config=dict(n_estimators=150, min_samples_leaf=5, max_features="sqrt"),
        build=_rf, history=None,
    ),
    "hgb": dict(
        family="Boosting (histogram gradient boosting)",
        config=dict(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, early_stopping=True,
                    validation_fraction=0.1, n_iter_no_change=15, scoring="loss"),
        build=_hgb, history=_hgb_history,
    ),
    "mlp": dict(
        family="Neural network (MLP)",
        config=dict(hidden_layer_sizes=(64, 32), early_stopping=True, validation_fraction=0.1,
                    n_iter_no_change=8, max_iter=60, batch_size=512),
        build=_mlp, history=_mlp_history,
    ),
    # Koushik: add "lr" (LogisticRegression in a StandardScaler pipeline) and
    # "gnb" (GaussianNB) here, same shape as the entries above.
}

# The Bayesian-tuned boosting model appears once scripts/bayes_opt.py has written its result.
_BEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "bayes_best_hgb.json")
if os.path.exists(_BEST):
    with open(_BEST) as _fh:
        _tuned = json.load(_fh)["config"]
    REGISTRY["hgb_tuned"] = dict(
        family="Boosting (histogram gradient boosting, Bayesian-tuned)",
        config=dict(**_tuned, early_stopping=True, validation_fraction=0.1, n_iter_no_change=15, scoring="loss"),
        build=_hgb, history=_hgb_history,
    )
