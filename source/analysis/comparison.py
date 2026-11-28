"""Reproduce the shared benchmark, nested CV and split-conformal evaluation."""
import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor

from ..evaluation import test_predictions
from ..modeling import FEATURES, TARGET, load_penguins, split_penguins
from ..models import m1
from ..ridge_evaluation import paired_mae_bootstrap, scores
from .uncertainty import (
    LEVELS, MODULES, fit_calibrated, fitting_procedure, interval_summary, proper_calibration_split,
)


def selected_parameter(estimator):
    params = getattr(estimator, "best_params_", {})
    if len(params) > 1:
        return {key.rsplit("__", 1)[-1]: value for key, value in params.items()}
    return next(iter(params.values()), None)


def conformal_test(calibrated, test):
    rows = []
    for name, model in calibrated.items():
        for level in LEVELS:
            prediction, lower, upper = model.interval(test[FEATURES], level)
            masks = {"all": np.ones(len(test), dtype=bool)}
            for feature in ("species", "sex"):
                labels = test[feature].fillna("<thiếu>")
                masks.update({f"{feature}={label}": labels.eq(label) for label in labels.unique()})
            for group, mask in masks.items():
                rows.append({"model": name, "level": level, "group": group,
                             **interval_summary(test.loc[mask, TARGET], lower[mask], upper[mask])})
    return rows


def benchmark():
    train, test = split_penguins(load_penguins())
    if not (train.index.is_unique and test.index.is_unique and train.index.intersection(test.index).empty):
        raise ValueError("Train and test row IDs must be unique and disjoint.")
    full = {name: fitting_procedure(name).fit(train[FEATURES], train[TARGET]) for name in MODULES}
    parameters = {name: selected_parameter(model) for name, model in full.items()}
    predictions = test_predictions(full, test)
    test_metrics = {name: scores(test[TARGET], pred) for name, pred in predictions.items()}
    intervals = {}
    for name in predictions.columns.drop("OLS"):
        ci = paired_mae_bootstrap(test[TARGET], predictions["OLS"], predictions[name]).iloc[0]
        intervals[name] = ci[["delta_g", "ci_low_g", "ci_high_g"]].to_dict()
    for strategy in ("mean", "median"):
        dummy = DummyRegressor(strategy=strategy).fit(train[FEATURES], train[TARGET])
        test_metrics[strategy] = scores(test[TARGET], dummy.predict(test[FEATURES]))

    rows = []
    fold_ids = []
    for fold, (fit_pos, val_pos) in enumerate(m1.cv_splitter().split(train), 1):
        fit, val = train.iloc[fit_pos], train.iloc[val_pos]
        fold_ids.append(val.index.tolist())
        for name, estimator in full.items():
            fixed = clone(getattr(estimator, "best_estimator_", estimator)).fit(fit[FEATURES], fit[TARGET])
            nested = fixed if name == "OLS" else fitting_procedure(name).fit(fit[FEATURES], fit[TARGET])
            for protocol, model, parameter in (("fixed", fixed, parameters[name]),
                                                ("nested", nested, selected_parameter(nested))):
                rows.append({"protocol": protocol, "fold": fold, "model": name, "parameter": parameter,
                             **scores(val[TARGET], model.predict(val[FEATURES]))})
    proper, cal = proper_calibration_split(train)
    calibrated = fit_calibrated(proper, cal, MODULES)
    conformal_metrics = {
        name: {"parameter": selected_parameter(model.predictor),
               **scores(test[TARGET], model.predictor.predict(test[FEATURES]))}
        for name, model in calibrated.items()
    }
    return {
        "row_ids": {"test": test.index.tolist(), "proper_train": proper.index.tolist(),
                    "calibration": cal.index.tolist(), "outer_validation": fold_ids},
        "parameters": parameters, "benchmark": test_metrics, "paired_delta_mae": intervals,
        "test_predictions": predictions.rename_axis("row_id").reset_index().to_dict("records"),
        "cv_folds": rows, "conformal_metrics": conformal_metrics,
        "conformal_intervals": conformal_test(calibrated, test),
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] / "report/comparison.json")
    output = parser.parse_args().output
    result = benchmark()
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(pd.DataFrame(result["benchmark"]).T.round(3))
    metrics = list(result["benchmark"]["OLS"])
    print(pd.DataFrame(result["cv_folds"]).groupby(["protocol", "model"])[metrics].agg(["mean", "std"]).round(3))
    print(pd.DataFrame(result["conformal_intervals"]).query("group == 'all'").round(3))


if __name__ == "__main__":
    main()
