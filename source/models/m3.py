from argparse import ArgumentParser
from warnings import warn

import numpy as np
from scipy.interpolate import make_smoothing_spline
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.utils.validation import check_is_fitted, validate_data

from ..modeling import (
    CATEGORICAL,
    FEATURES,
    TARGET,
    load_penguins,
    make_pipeline,
    run_model,
    split_penguins,
)

NAME = "M3"
SMOOTH_FEATURES = ("flipper_length_mm", "bill_length_mm", "bill_depth_mm")
LAMBDA_CANDIDATES = (0.1, 1.0, 10.0, 100.0)


class AdditiveSmoothingSpline(RegressorMixin, BaseEstimator):
    """Fit centered natural cubic smoothing splines and linear effects by backfitting."""

    def __init__(self, *, smooth_indices, lam, max_iter=300, tol=0.01):
        self.smooth_indices = smooth_indices
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        X, y = validate_data(self, X, y, y_numeric=True)
        smooth_indices = tuple(self.smooth_indices)
        if not smooth_indices:
            raise ValueError("At least one smooth feature is required.")
        if not np.isfinite(self.lam) or self.lam < 0:
            raise ValueError("lam must be finite and nonnegative.")
        if self.max_iter < 1 or not np.isfinite(self.tol) or self.tol <= 0:
            raise ValueError("max_iter and tol must be positive, with finite tol.")

        # The shared preprocessor emits numeric columns before the one-hot columns.
        linear_indices = np.delete(np.arange(X.shape[1]), smooth_indices)
        linear_design = np.column_stack((np.ones(len(X)), X[:, linear_indices]))
        groups = [
            np.unique(X[:, index], return_inverse=True, return_counts=True)
            for index in smooth_indices
        ]
        if any(len(knots) < 5 for knots, _, _ in groups):
            raise ValueError("Each smooth feature needs at least five distinct training values.")

        components = np.zeros((len(X), len(groups)))
        linear_fitted = np.full(len(X), y.mean())
        for n_iter in range(1, self.max_iter + 1):
            previous_components = components.copy()
            previous_linear = linear_fitted
            splines = []
            centers = []
            residual = y - linear_fitted - components.sum(axis=1)
            for term, (knots, inverse, counts) in enumerate(groups):
                residual += components[:, term]
                # Counts preserve the original squared-error objective for repeated x.
                means = np.bincount(inverse, weights=residual) / counts
                spline = make_smoothing_spline(knots, means, w=counts, lam=self.lam)
                fitted = spline(knots)[inverse]
                center = fitted.mean()
                components[:, term] = fitted - center
                residual -= components[:, term]
                splines.append(spline)
                centers.append(center)

            coefficients = np.linalg.lstsq(linear_design, y - components.sum(axis=1), rcond=None)[0]
            linear_fitted = linear_design @ coefficients
            change = max(
                np.abs(components - previous_components).max(),
                np.abs(linear_fitted - previous_linear).max(),
            )
            if change <= self.tol:
                break
        else:
            warn("Additive smoothing spline backfitting reached max_iter.", ConvergenceWarning, stacklevel=2)

        self.n_iter_ = n_iter
        self.intercept_ = coefficients[0]
        self.linear_indices_ = linear_indices
        self.linear_coef_ = coefficients[1:]
        self.smooth_indices_ = smooth_indices
        self.splines_ = splines
        self.centers_ = centers
        return self

    def predict(self, X):
        check_is_fitted(self, "splines_")
        X = validate_data(self, X, reset=False)
        predictions = self.intercept_ + X[:, self.linear_indices_] @ self.linear_coef_
        for index, spline, center in zip(self.smooth_indices_, self.splines_, self.centers_):
            x = X[:, index]
            left, right = spline.t[spline.k], spline.t[-spline.k - 1]
            inside = np.clip(x, left, right)
            # Extend each natural spline with its endpoint tangents beyond the training range.
            predictions += spline(inside) + (x - inside) * spline(inside, nu=1) - center
        return predictions


NUMERIC_FEATURES = [feature for feature in FEATURES if feature not in CATEGORICAL]
ESTIMATOR = AdditiveSmoothingSpline(
    smooth_indices=tuple(
        index for index, feature in enumerate(NUMERIC_FEATURES) if feature in SMOOTH_FEATURES
    ),
    lam=100.0,
)


def build_model():
    return make_pipeline(ESTIMATOR)


def select_lambda():
    train, _ = split_penguins(load_penguins())
    search = GridSearchCV(
        build_model(),
        {"regression__lam": LAMBDA_CANDIDATES},
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="neg_mean_absolute_error",
        refit=False,
        error_score="raise",
    )
    search.fit(train[FEATURES], train[TARGET])
    for lam, score in zip(LAMBDA_CANDIDATES, search.cv_results_["mean_test_score"]):
        print(f"λ = {lam:g} | MAE CV (g) = {-score:.3f}")
    print(f"λ được chọn = {search.best_params_['regression__lam']:g}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Huấn luyện và đánh giá natural cubic smoothing spline M3.")
    parser.add_argument("--select-lam", action="store_true", help="Chọn λ bằng cross-validation trên train.")
    if parser.parse_args().select_lam:
        select_lambda()
    else:
        run_model(NAME, build_model(), is_example=False)
