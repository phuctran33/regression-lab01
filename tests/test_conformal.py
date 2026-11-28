import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from source.analysis.uncertainty import (
    conformal_radius, fit_calibrated, interval_summary, proper_calibration_split,
)
from source.modeling import FEATURES, TARGET, load_penguins, split_penguins
from source.models import ols
from source.prediction_app import PredictionApp


class ConformalTests(unittest.TestCase):
    def test_order_statistics_and_small_calibration(self):
        scores = np.arange(55, 0, -1)
        self.assertEqual(conformal_radius(scores, 0.90), 51)
        self.assertEqual(conformal_radius(scores, 0.95), 54)
        self.assertTrue(np.isinf(conformal_radius([1, 2, 3], 0.95)))
        with self.assertRaises(ValueError):
            conformal_radius([np.nan], 0.95)
        summary = interval_summary(np.array([0, 1, 2]), np.array([0, 0, 0]), np.array([1, 1, 1]))
        self.assertEqual(summary["coverage"], 2 / 3)
        self.assertEqual(summary["outside"], 1)
        self.assertEqual(summary["width_g"], 1)

    def test_all_calibration_and_inner_fits_use_only_proper_training(self):
        train, test = split_penguins(load_penguins())
        proper, cal = proper_calibration_split(train)
        self.assertEqual((len(proper), len(cal), len(test)), (218, 55, 69))
        self.assertEqual(set(proper.index) | set(cal.index), set(train.index))
        self.assertTrue(test.index.intersection(train.index).empty)
        original = Pipeline.fit
        fit_sizes = set()

        def track(pipeline, X, y=None, **kwargs):
            if "regression" in pipeline.named_steps:
                self.assertTrue(X.index.isin(proper.index).all())
                self.assertTrue(X.index.equals(y.index))
                fit_sizes.add(len(X))
            return original(pipeline, X, y, **kwargs)

        with patch.object(Pipeline, "fit", track):
            models = fit_calibrated(proper, cal, ["OLS", "M1", "M2", "M3"])
        self.assertEqual(fit_sizes, {174, 175, 218})
        for model in models.values():
            scores = np.sort(abs(cal[TARGET] - model.predictor.predict(cal[FEATURES])))
            for level, rank in ((.9, 51), (.95, 54)):
                point, lower, upper = model.interval(test[FEATURES], level)
                np.testing.assert_allclose(point, model.predictor.predict(test[FEATURES]))
                np.testing.assert_allclose(point - lower, scores[rank - 1])
                np.testing.assert_allclose(upper - point, scores[rank - 1])
        with self.assertRaises(ValueError):
            fit_calibrated(train, cal, ["OLS"])

    def test_widget_changes_point_and_interval_together(self):
        train, test = split_penguins(load_penguins())
        models = {"OLS": ols.build_model().fit(train[FEATURES], train[TARGET])}
        app = PredictionApp(models, train, {TARGET: "Khối lượng (g)"}, test=test)
        inputs = pd.DataFrame([{f: c.value for f, c in app.controls.items()}])
        with patch.object(Pipeline, "fit", side_effect=AssertionError("Unexpected refit")):
            for level in (.9, .95):
                app.level.value = level
                app.predict_button.click()
                for values in app.calibrated["OLS"].interval(inputs, level):
                    self.assertIn(f"{values[0]:,.1f}", app.result.value)
                self.assertIn("69 mẫu test", app.result.value)
                self.assertTrue(bytes(app.chart.value).startswith(b"\x89PNG"))
            app.protocol.value = "full"
            app.predict_button.click()
            self.assertIn(f"{models['OLS'].predict(inputs)[0]:,.1f}", app.result.value)
            self.assertIn("CI 95%", app.result.value)
            self.assertTrue(app.level.disabled)
