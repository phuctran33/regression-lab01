"""Numerical and state regressions for the notebook evidence panels."""
import csv
from html import unescape
from io import StringIO
import re
from urllib.parse import unquote
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from source import prediction_evidence as ev
from source.analysis.uncertainty import LEVELS, MODULES, fitting_procedure
from source.modeling import FEATURES, TARGET, load_penguins, split_penguins
from source.prediction_app import PredictionApp
from source.prediction_figures import number


class PredictionEvidenceTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.test = split_penguins(load_penguins())
        models = {n: fitting_procedure(n).fit(cls.train[FEATURES], cls.train[TARGET]) for n in MODULES}
        cls.app = PredictionApp(models, cls.train, {}, test=cls.test)

    def setUp(self):
        self.app._reset()

    def test_csv_uses_the_current_fit_and_only_its_intervals(self):
        app = self.app
        for mode in ("conformal", "full"):
            app.protocol.value = mode
            for name in MODULES:
                app.model_picker.value = name
                app.predict_button.click()
                uri = re.search(r'href="data:text/csv;charset=utf-8,([^"]+)"', app.result.value)
                self.assertIsNotNone(uri)
                rows = list(csv.DictReader(StringIO(unquote(unescape(uri[1])).lstrip("\ufeff"))))
                self.assertEqual(len(rows), 1)
                row = rows[0]
                self.assertEqual((row["model"], row["protocol"]), (name, mode))
                for key in FEATURES:
                    self.assertEqual(str(app.defaults[key]), row[key])
                predictor = app.models[name] if mode == "full" else app.calibrated[name].predictor
                point = predictor.predict(pd.DataFrame([app.defaults]))[0]
                self.assertAlmostEqual(float(row["prediction_g"]), point)
                if mode == "conformal":
                    _, low, high = app.calibrated[name].interval(pd.DataFrame([app.defaults]), app.level.value)
                    np.testing.assert_allclose([float(row["lower_g"]), float(row["upper_g"])], [low[0], high[0]])
                    self.assertAlmostEqual(float(row["empirical_coverage"]), app.performance[name, app.level.value]["coverage"])
                elif name == "OLS":
                    self.assertLess(float(row["lower_g"]), float(row["mean_ci_lower_g"]))
                    self.assertGreater(float(row["upper_g"]), float(row["mean_ci_upper_g"]))
                else:
                    self.assertEqual([row[k] for k in ("lower_g", "mean_ci_lower_g", "empirical_coverage")], [""]*3)
        app.controls["flipper_length_mm"].value += 1
        self.assertNotIn("data:text/csv", app.result.value)

    def test_inspector_tracks_test_ids_and_calibration_rank_in_both_layouts(self):
        app = self.app
        app.model_picker.value = "M2"
        app.level.value = .9
        app.predict_button.click()
        app.section.value = "verify"
        inspection = app.chart.inspection
        expected = ev.test_rows(self.test, app.predictions["conformal", "M2"], app.calibrated["M2"].radii[.9])
        for (idx, row), (title, detail) in zip(expected.iterrows(), inspection["labels"], strict=True):
            self.assertEqual(title, f"M2 · ID {idx} · {row.species}")
            self.assertIn(number(row.actual), detail)
            self.assertIn(number(row.low), detail)
            self.assertIn("Trong khoảng" if row.inside else "Ngoài khoảng", detail)
        for layout in ("wide", "compact"):
            positions = np.asarray(inspection[layout])
            self.assertEqual(positions.shape, (69, 2))
            self.assertTrue(((0 <= positions) & (positions <= 1)).all())
        app.section.value = "method"
        self.assertEqual(app.chart.inspection["selected"], 50)
        self.assertIn("hạng 51/55", app.chart.inspection["labels"][50][0])

    def test_intervals_and_coverage_width_match_the_calibrated_predictors(self):
        app = self.app
        point = pd.DataFrame([app.defaults])
        for level in LEVELS:
            compared = ev.comparison_rows(app.calibrated, app.performance, point, level)
            self.assertEqual(list(compared.index), list(MODULES))
            for name, fitted in app.calibrated.items():
                pred, low, high = fitted.interval(point, level)
                row = compared.loc[name]
                np.testing.assert_allclose([row.pred, row.low, row.high, row.width],
                                           [pred[0], low[0], high[0], high[0]-low[0]])
                sorted_rows = ev.test_rows(self.test, app.predictions["conformal", name], fitted.radii[level])
                pred, low, high = fitted.interval(self.test[FEATURES], level)
                self.assertEqual(set(sorted_rows.index), set(self.test.index))
                self.assertTrue(sorted_rows.pred.is_monotonic_increasing)
                original_order = sorted_rows.loc[self.test.index]
                np.testing.assert_allclose(original_order.pred, pred)
                np.testing.assert_allclose(original_order.low, low)
                np.testing.assert_allclose(original_order.high, high)
                np.testing.assert_array_equal(original_order.inside, (low <= self.test[TARGET]) & (self.test[TARGET] <= high))
                self.assertAlmostEqual(row.coverage, sorted_rows.inside.mean())
                self.assertAlmostEqual(row.mae, sorted_rows.residual.abs().mean())
        edge = ev.test_rows(pd.DataFrame({TARGET:[90., 110., 110.1]}), [100.]*3, 10.)
        self.assertEqual(edge.inside.tolist(), [True, True, False])

    def test_preset_reset_and_navigation_cannot_refit_or_show_stale_data(self):
        app = self.app
        with patch.object(Pipeline, "fit", side_effect=AssertionError("Refit from UI")):
            app.model_picker.value = "M2"
            app.level.value = .9
            for label, idx in app.presets.items():
                app.preset.value = label
                self.assertIn(idx, self.train.index)
                self.assertNotIn(idx, self.test.index)
                self.assertEqual({k:c.value for k,c in app.controls.items()}, self.train.loc[idx, FEATURES].to_dict())
                self.assertFalse(app.rendered)
                self.assertFalse(app.chart.value)
                self.assertNotIn("pg-answer", app.result.value)
            app.predict_button.click()
            expected = (app.model_picker.value, app.level.value, app.protocol.value, app.preset.value)
            for section in ("compare", "verify", "method", "predict"):
                app.section.value = section
                self.assertTrue(app.rendered)
                self.assertTrue(app.chart.value.startswith(b"\x89PNG"))
                self.assertEqual(expected, (app.model_picker.value, app.level.value, app.protocol.value, app.preset.value))
            app.controls["bill_length_mm"].valid = False
            self.assertTrue(app.predict_button.disabled)
            app._reset()
            self.assertEqual({k:c.value for k,c in app.controls.items()}, app.defaults)
            self.assertTrue(all(app.controls[k].valid for k in app.numeric_features))
            self.assertEqual((app.section.value,app.view.value,app.protocol.value,app.level.value,app.preset.index),
                             ("predict","trend","conformal",.95,0))
            self.assertIn("2.935,5", app.result.announcement)
            self.assertNotIn("pg-warning", app.result.value)

    def test_actual_errors_and_calibration_notes_follow_the_active_fit(self):
        app = self.app
        app.model_picker.value = "M2"
        for mode, predictor in [("full", app.models["M2"]), ("conformal", app.calibrated["M2"].predictor)]:
            app.protocol.value = mode
            app.predict_button.click()
            app.view.value = "actual"
            expected = predictor.predict(self.test[FEATURES])
            np.testing.assert_allclose(app.predictions[mode, "M2"], expected)
            absolute = np.abs(self.test[TARGET] - expected)
            for text in [str(absolute.idxmax()), number(absolute.max()), number(absolute.mean())]:
                self.assertIn(text, app.chart.description)
        app.section.value = "method"
        for level,k in ((.9,51),(.95,54)):
            app.level.value = level
            app.predict_button.click()
            self.assertIn(f"hạng {k}/55",app.chart.description)
            self.assertAlmostEqual(sorted(app.calibration_scores["M2"])[k-1], app.calibrated["M2"].radii[level])
        app.section.value = "verify"
        app.reliability.value = "width"
        self.assertIn("7 và 1", app.chart.description)
        app.section.value = "compare"
        app.protocol.value = "full"
        app.predict_button.click()
        self.assertIn("4/4", app.chart.description)
        self.assertIn("Nested CV",app.caption.value)
        self.assertIsNone(ev.verified_benchmark(self.test.iloc[::-1], {"OLS":app.predictions["full","OLS"]}))
