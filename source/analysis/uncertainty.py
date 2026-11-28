"""Fit split-conformal predictors on proper training only, with a shared calibration split."""
from dataclasses import dataclass
from math import ceil

import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split

from ..modeling import FEATURES, TARGET
from ..models import m1, m2, m3, ols

MODULES = {module.NAME: module for module in (ols, m1, m2, m3)}
LEVELS = (0.90, 0.95)


def fitting_procedure(name):
    """Unfitted selection procedure; all inner fits include preprocessing."""
    model = MODULES[name].build_model()
    if name in ("OLS", "M1"):
        return model
    grid = ({"preprocess__numeric__scaler__n_knots": m2.KNOT_GRID} if name == "M2"
            else {"regression__lam": m3.LAMBDA_CANDIDATES})
    return GridSearchCV(model, grid, cv=m1.cv_splitter(), scoring="neg_mean_absolute_error",
                        error_score="raise")


def proper_calibration_split(train):
    if not train.index.is_unique:
        raise ValueError("Training row IDs must be unique.")
    return train_test_split(train, test_size=0.20, random_state=m1.SEED)


def conformal_radius(scores, level):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Calibration requires finite absolute residuals.")
    if not 0 < level < 1:
        raise ValueError("Coverage must be strictly between zero and one.")
    k = ceil((len(scores) + 1) * level)
    return np.inf if k > len(scores) else np.partition(scores, k - 1)[k - 1]


@dataclass
class CalibratedModel:
    predictor: object
    radii: dict

    def interval(self, features, level):
        point = self.predictor.predict(features)
        radius = self.radii[level]
        return point, point - radius, point + radius


def fit_calibrated(proper, calibration, names):
    if (not proper.index.is_unique or not calibration.index.is_unique
            or not proper.index.intersection(calibration.index).empty):
        raise ValueError("Proper training and calibration must have distinct row IDs.")
    result = {}
    for name in names:
        procedure = fitting_procedure(name).fit(proper[FEATURES], proper[TARGET])
        scores = np.abs(calibration[TARGET] - procedure.predict(calibration[FEATURES]))
        radii = {level: conformal_radius(scores, level) for level in LEVELS}
        result[name] = CalibratedModel(procedure, radii)
    return result


def interval_summary(actual, lower, upper):
    actual, lower, upper = (np.asarray(v, dtype=float) for v in (actual, lower, upper))
    if actual.ndim != 1 or not actual.size or actual.shape != lower.shape or actual.shape != upper.shape:
        raise ValueError("Each observed target must have one interval.")
    if not np.isfinite(actual).all() or np.isnan(lower).any() or np.isnan(upper).any() or (lower > upper).any():
        raise ValueError("Invalid bounds or observed target.")
    covered = (actual >= lower) & (actual <= upper)
    return {"n": len(actual), "outside": int((~covered).sum()), "coverage": float(covered.mean()),
            "width_g": float(np.mean(upper - lower))}
