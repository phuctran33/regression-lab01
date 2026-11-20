import unittest

import numpy as np
from scipy.interpolate import make_smoothing_spline

from source.modeling import CATEGORICAL, FEATURES, TARGET, load_penguins, split_penguins
from source.models import m3, ols


class RegressionTests(unittest.TestCase):
    def test_repeated_measurements_and_natural_boundary(self):
        knots = np.arange(6, dtype=float)
        means = np.array([0, 2, 1, 3, 1, 4], dtype=float)
        counts = np.array([1, 3, 1, 2, 1, 4])
        x = np.repeat(knots, counts)[::-1]
        y = np.repeat(means, counts)[::-1] + np.linspace(-1, 1, counts.sum())
        grouped_means = [y[x == knot].mean() for knot in knots]
        expected = make_smoothing_spline(knots, grouped_means, w=counts, lam=0.1)

        model = m3.AdditiveSmoothingSpline(smooth_indices=(0,), lam=0.1).fit(x[:, None], y)
        grid = np.linspace(knots[0], knots[-1], 25)
        np.testing.assert_allclose(model.predict(grid[:, None]), expected(grid), atol=1e-10)
        np.testing.assert_allclose(model.splines_[0](knots[[0, -1]], nu=2), 0, atol=1e-10)

        tails = np.array([-2, -1, 0, 5, 6, 7])[:, None]
        predictions = model.predict(tails).reshape(2, 3)
        np.testing.assert_allclose(np.diff(predictions, n=2, axis=1), 0, atol=1e-10)

    def test_shared_pipeline_fits_on_train_and_predicts_holdout(self):
        train, test = split_penguins(load_penguins())
        numeric_features = [feature for feature in FEATURES if feature not in CATEGORICAL]
        expected_means = train[numeric_features].fillna(train[numeric_features].median()).mean()

        for module in (ols, m3):
            with self.subTest(model=module.NAME):
                model = module.build_model().fit(train[FEATURES], train[TARGET])
                preprocessor = model.named_steps["preprocess"]
                scaler = preprocessor.named_transformers_["numeric"].named_steps["scaler"]
                np.testing.assert_allclose(scaler.mean_, expected_means)
                predictions = model.predict(test[FEATURES])
                self.assertEqual(predictions.shape, (len(test),))
                self.assertTrue(np.isfinite(predictions).all())

                if module is m3:
                    regression = model.named_steps["regression"]
                    self.assertIsNotNone(m3.ESTIMATOR)
                    self.assertLess(regression.n_iter_, regression.max_iter)
                    x_train = preprocessor.transform(train[FEATURES])
                    residual = train[TARGET].to_numpy() - regression.predict(x_train)
                    linear_design = x_train[:, regression.linear_indices_]
                    np.testing.assert_allclose(linear_design.T @ residual, 0, atol=1e-8)
                    for index, spline, center in zip(
                        regression.smooth_indices_, regression.splines_, regression.centers_, strict=True
                    ):
                        self.assertAlmostEqual((spline(x_train[:, index]) - center).mean(), 0)
