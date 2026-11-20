from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.pipeline import Pipeline

matplotlib.use("Agg")

from source.evaluation import analyze, fit_ols_inference, paired_bootstrap
from source.modeling import FEATURES, TARGET, load_penguins, split_penguins
from source.models import m3, ols
from source.prediction_app import PredictionApp


class EvaluationTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.test = split_penguins(load_penguins())

    def test_statsmodels_matches_fitted_sklearn_pipeline(self):
        model = ols.build_model().fit(self.train[FEATURES], self.train[TARGET])
        result, gap = fit_ols_inference(model, self.train)
        preprocessor = model.named_steps["preprocess"]
        self.assertEqual(result.model.exog_names, [
            "const", "flipper_length_mm", "bill_length_mm", "bill_depth_mm",
            "species_Chinstrap", "species_Gentoo", "sex_male",
        ])
        np.testing.assert_allclose(result.model.exog[:, 1:],
                                   preprocessor.transform(self.train[FEATURES]))
        self.assertTrue(result.fittedvalues.index.equals(self.train.index))
        self.assertLess(gap, 1e-8)
        regression = model.named_steps["regression"]
        np.testing.assert_allclose(result.params, [regression.intercept_, *regression.coef_])
        self.assertEqual(list(result.params.index), result.model.exog_names)
        design_test = np.column_stack((
            np.ones(len(self.test)), preprocessor.transform(self.test[FEATURES]),
        ))
        np.testing.assert_allclose(result.predict(design_test),
                                   model.predict(self.test[FEATURES]), rtol=0, atol=1e-8)

    def test_paired_bootstrap_is_reproducible_and_keeps_the_difference(self):
        errors = [1, -4, 8, 100, -200]
        residuals = pd.DataFrame({"OLS": np.abs(errors) + 3, "M3": errors})
        result = paired_bootstrap(residuals, seed=7, n_resamples=200)
        pd.testing.assert_frame_equal(result, paired_bootstrap(residuals, seed=7, n_resamples=200))
        np.testing.assert_allclose(result.loc["MAE"], [-3, -3, -3])

    def test_evaluation_fits_train_only_and_writes_tables_and_figures(self):
        original_fit = Pipeline.fit
        fit_indices = []

        def tracked_fit(pipeline, X, y=None, **kwargs):
            if "regression" in pipeline.named_steps:
                self.assertIsInstance(X, pd.DataFrame)
                self.assertTrue(X.index.isin(self.train.index).all())
                self.assertTrue(X.index.intersection(self.test.index).empty)
                self.assertTrue(X.index.equals(y.index))
                fit_indices.append(frozenset(X.index))
            return original_fit(pipeline, X, y, **kwargs)

        with TemporaryDirectory() as directory, patch.object(Pipeline, "fit", tracked_fit):
            figure_dir = Path(directory)
            tables = analyze(self.train, self.test, figure_dir)
            self.assertEqual(len(fit_indices), 12)
            self.assertEqual(fit_indices[2:7], fit_indices[7:12])
            self.assertTrue(all(len(indices) < len(self.train) for indices in fit_indices[2:]))
            self.assertEqual(tables["OLS coefficients"].shape, (7, 11))
            self.assertEqual(tables["Train CV, 5 folds"].shape, (8, 2))
            self.assertEqual(tables["Held-out test"].shape, (2, 4))
            self.assertEqual(tables["Paired test bootstrap"].shape, (2, 3))
            self.assertTrue(np.isfinite(tables["Paired test bootstrap"]).all().all())
            self.assertEqual(len(list(figure_dir.glob("*.png"))), 3)
            for path in figure_dir.glob("*.png"):
                with Image.open(path) as figure:
                    self.assertEqual(figure.format, "PNG")
                    self.assertGreater(figure.width, 500)
                    figure.verify()


class PredictionAppTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, _ = split_penguins(load_penguins())
        cls.models = {
            module.NAME: module.build_model().fit(cls.train[FEATURES], cls.train[TARGET])
            for module in (ols, m3)
        }

    def test_predict_button_updates_both_models_and_numeric_and_category_plots(self):
        app = PredictionApp(self.models, self.train, {TARGET: "Khối lượng cơ thể (g)"})
        self.assertEqual(list(app.controls), FEATURES)
        with patch.object(Pipeline, "fit", side_effect=AssertionError("Prediction refit a model")):
            app.feature_picker.value = "flipper_length_mm"
            app.controls["flipper_length_mm"].value = 190
            app.predict_button.click()
            first_image = bytes(app.chart.value)
            self.assertTrue(first_image.startswith(b"\x89PNG"))
            self.assertIn("OLS · body_mass_g", app.result.value)
            self.assertIn("M3", app.result.value)

            app.model_picker.value = "M3"
            app.feature_picker.value = "sex"
            app.controls["sex"].value = "male"
            self.assertEqual(len(app.chart.value), 0)
            app.predict_button.click()
            values = {feature: widget.value for feature, widget in app.controls.items()}
            expected = self.models["M3"].predict(pd.DataFrame([values]))[0]
            self.assertIn(f"M3 · body_mass_g = {expected:,.1f} g", app.result.value)
            self.assertTrue(bytes(app.chart.value).startswith(b"\x89PNG"))
            self.assertNotEqual(first_image, bytes(app.chart.value))
            app.predict_button.click()
            self.assertIn(f"{expected:,.1f} g", app.result.value)
            self.assertFalse(app.predict_button.disabled)

            with patch.object(self.models["M3"], "predict", side_effect=ValueError("bad input")):
                app.predict_button.click()
            self.assertIn("Không thể dự đoán", app.result.value)
            self.assertEqual(len(app.chart.value), 0)
            self.assertFalse(app.predict_button.disabled)
