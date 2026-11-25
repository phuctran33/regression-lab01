"""M1 Ridge: choose alpha on train; expose the fitted pipeline to the notebook."""

from argparse import ArgumentParser

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.utils.validation import check_is_fitted

from ..modeling import (
    FEATURES, TARGET, load_penguins, make_pipeline, regression_metrics, split_penguins,
)

NAME = "M1"
BASELINE_ALPHA = 1.0
ESTIMATOR = Ridge(alpha=BASELINE_ALPHA)
ALPHA_CANDIDATES = (0.01, 0.1, 1.0, 10.0, 100.0)
SEED = 42


def cv_splitter():
    return KFold(n_splits=5, shuffle=True, random_state=SEED)


class RidgeSearchCV(GridSearchCV):
    """GridSearchCV with the fitted named_steps interface used by the notebook."""

    @property
    def named_steps(self):
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.named_steps


def build_model():
    """Return an unfitted search; preprocessing is refit in each training fold."""
    return RidgeSearchCV(
        make_pipeline(ESTIMATOR), {"regression__alpha": ALPHA_CANDIDATES},
        scoring="neg_mean_absolute_error", cv=cv_splitter(), refit=True,
        n_jobs=1, error_score="raise", verbose=0,
    )


def main():
    parser = ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--baseline", action="store_true", help="Reproduce week 1 alpha=1.")
    mode.add_argument("--select-alpha", action="store_true", help="Train CV only; no test metrics.")
    args = parser.parse_args()
    train, test = split_penguins(load_penguins())
    model = make_pipeline(ESTIMATOR) if args.baseline else build_model()
    model.fit(train[FEATURES], train[TARGET])
    print(f"M1 | Ridge | {'week 1 baseline' if args.baseline else 'train-selected alpha'}")
    if not args.baseline:
        for i, params in enumerate(model.cv_results_["params"]):
            scores = -np.array([model.cv_results_[f"split{k}_test_score"][i] for k in range(5)])
            if args.select_alpha:
                print(f"alpha={params['regression__alpha']:g} | CV MAE={scores.mean():.6f} +/- {scores.std(ddof=1):.6f} g")
        print(f"Selected alpha: {model.best_params_['regression__alpha']:g}")
        print(f"Development CV MAE (g): {-model.best_score_:.6f}")
    if args.select_alpha:
        return
    predicted = model.predict(test[FEATURES])
    metrics = regression_metrics(test[TARGET], predicted)
    metrics["MSE (g²)"] = mean_squared_error(test[TARGET], predicted)
    for name in ("MAE (g)", "MSE (g²)", "RMSE (g)", "R²"):
        print(f"Test {name}: {metrics[name]:.6f}")


if __name__ == "__main__":
    main()
