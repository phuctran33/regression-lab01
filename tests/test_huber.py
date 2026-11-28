import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

from source.huber_evaluation import evaluate_huber, paired_mae_interval
from source.modeling import FEATURES, TARGET, load_penguins, split_penguins
from source.models import m4


class HuberEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.test = split_penguins(load_penguins())
        cls.tables, cls.predictions = evaluate_huber(cls.train, cls.test)

    def test_grid_selects_the_pipeline_model(self):
        grid = self.tables["Lưới siêu tham số trên train"]
        self.assertEqual(len(grid), 48)
        estimator = m4.build_model().named_steps["regression"]
        self.assertIsInstance(estimator, HuberRegressor)
        self.assertEqual((estimator.epsilon, estimator.alpha), tuple(grid.loc[0, ["epsilon", "alpha"]]))
        grid_mae = grid.loc[0, ["MAE TB (g)", "MAE SD (g)"]].to_numpy(dtype=float)
        cv_mae = self.tables["CV 5 fold trên train"].loc[("M4", "MAE (g)")].to_numpy()
        np.testing.assert_allclose(grid_mae, cv_mae)

    def test_holdout_uses_the_runtime_model_and_preserves_rows(self):
        pd.testing.assert_index_equal(self.predictions.index, self.test.index)
        model = m4.build_model().fit(self.train[FEATURES], self.train[TARGET])
        np.testing.assert_allclose(self.predictions["M4"], model.predict(self.test[FEATURES]))
        metrics = self.tables["Chỉ số trên test"]
        np.testing.assert_allclose(metrics["RMSE (g)"].pow(2), metrics["MSE (g²)"])
        subgroup = self.tables["Sai số test theo loài và giới tính"]
        self.assertTrue(subgroup["n"].groupby(level=["Biến", "Mô hình"]).sum().eq(len(self.test)).all())
        self.assertIn("<thiếu>", subgroup.index.get_level_values("Nhóm"))

    def test_sensitivity_reuses_train_points_with_the_same_removals(self):
        sensitivity = self.tables["Độ nhạy khi fit lại trên train"]
        for _, records in sensitivity.groupby(level="Kịch bản"):
            self.assertEqual(set(records.index.get_level_values("Mô hình")), {"M4", "OLS"})
            self.assertEqual(records["CSV bỏ khi fit"].nunique(), 1)
            removed = [int(index) for index in records["CSV bỏ khi fit"].iloc[0].split(", ")]
            self.assertEqual(self.train.index.isin(removed).sum(), len(removed))
            self.assertTrue(records["n còn lại"].eq(len(self.train) - len(removed)).all())
            self.assertTrue(records["|Δ dự đoán| max (g)"].ge(records["|Δ dự đoán| TB (g)"]).all())

    def test_bootstrap_keeps_errors_paired(self):
        predictions = pd.DataFrame({
            TARGET: [100, 200, 300],
            "OLS": [105, 215, 275],
            "M4": [97, 213, 277],
        })
        result = paired_mae_interval(predictions)
        self.assertEqual(result["n"], 3)
        np.testing.assert_allclose(result.drop("n"), [-2, -2, -2])


if __name__ == "__main__":
    unittest.main()
