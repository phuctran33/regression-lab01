from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from source.modeling import FEATURES, TARGET, load_penguins, split_penguins
from source.models import m1
from source.ridge_evaluation import (
    analyze, build_ols_reference, paired_mae_bootstrap, subgroup_errors,
    validate_pairs, write_outputs,
)


class RidgeWeek2Tests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.test = split_penguins(load_penguins())
        cls.tables = analyze(cls.train, cls.test)

    def test_search_uses_protocol_candidates_and_silently_refits_selected_pipeline(self):
        output = StringIO()
        with redirect_stdout(output):
            model = m1.build_model().fit(self.train[FEATURES], self.train[TARGET])
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(tuple(model.param_grid['regression__alpha']), (0.01, 0.1, 1, 10, 100))
        self.assertEqual(model.best_score_, max(model.cv_results_['mean_test_score']))
        self.assertEqual(model.named_steps['regression'].alpha, model.best_params_['regression__alpha'])
        self.assertFalse(hasattr(m1.ESTIMATOR, 'coef_'))
        self.assertFalse(hasattr(clone(model), 'best_estimator_'))
        np.testing.assert_allclose(model.predict(self.test[FEATURES]), model.best_estimator_.predict(self.test[FEATURES]))
        self.assertEqual(len(model.named_steps['preprocess'].get_feature_names_out()), len(model.named_steps['regression'].coef_))

    def test_all_fitting_is_train_only_and_partial_fits_use_the_same_five_fold_sets(self):
        original_fit = Pipeline.fit
        fit_ids = []
        allowed = set(self.train.index)
        expected = {frozenset(self.train.index[positions]) for positions, _ in m1.cv_splitter().split(self.train)}

        def record(pipeline, X, y=None, **params):
            if 'preprocess' in pipeline.named_steps:
                self.assertTrue(X.index.equals(y.index))
                self.assertTrue(set(X.index).issubset(allowed))
                self.assertTrue(X.index.intersection(self.test.index).empty)
                fit_ids.append(frozenset(X.index))
            return original_fit(pipeline, X, y, **params)

        with patch.object(Pipeline, 'fit', record):
            tables = analyze(self.train, self.test)
        partial = {ids for ids in fit_ids if len(ids) < len(self.train)}
        self.assertEqual(partial, expected)
        self.assertEqual(set(tables['cv_folds'].model), {'OLS', 'M1'})
        self.assertEqual(tables['cv_folds'].groupby('model').size().tolist(), [5, 5])

    def test_paired_bootstrap_preserves_pairs_and_is_reproducible(self):
        actual = pd.Series([10., 20., 30., 40.], index=[7, 2, 11, 9])
        ols = actual - [5., 6., 7., 8.]
        ridge = actual - [2., 3., 4., 5.]
        first = paired_mae_bootstrap(actual, ols, ridge, seed=7, n_resamples=200)
        pd.testing.assert_frame_equal(first, paired_mae_bootstrap(actual, ols, ridge, seed=7, n_resamples=200))
        np.testing.assert_allclose(first[['delta_g', 'ci_low_g', 'ci_high_g']], [[-3, -3, -3]])

    def test_pair_validation_rejects_shuffled_ids_and_nonfinite_values(self):
        actual = pd.Series([1., 2., 3.], index=[4, 1, 8])
        with self.assertRaisesRegex(ValueError, 'row IDs'):
            validate_pairs(actual, actual.iloc[::-1], actual)
        with self.assertRaisesRegex(ValueError, 'finite'):
            validate_pairs(actual, actual * np.nan, actual)
        overlapping = self.test.copy()
        overlapping.index = list(self.train.index[:len(overlapping)])
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            analyze(self.train, overlapping)

    def test_group_metrics_include_missing_sex_and_keep_residual_direction(self):
        data = pd.DataFrame({'species': ['A', 'A', 'B'], 'sex': ['male', None, None], TARGET: [10., 20., 30.]}, index=[9, 2, 5])
        predictions = pd.DataFrame({'OLS': [8., 18., 28.], 'M1': [7., 17., 27.]}, index=data.index)
        groups = subgroup_errors(data, predictions)
        missing = groups.query("feature == 'sex' and level == '<missing>' and model == 'M1'").iloc[0]
        self.assertEqual(missing['n'], 2)
        self.assertEqual(missing['MAE (g)'], 3)
        self.assertEqual(missing['Mean residual (g)'], 3)
        for _, group in groups.groupby(['feature', 'model']):
            self.assertEqual(group['n'].sum(), len(data))

    def test_cv_uses_sample_sd_and_test_mse_equals_rmse_squared(self):
        for row in self.tables['cv_summary'].itertuples(index=False):
            values = self.tables['cv_folds'].query('model == @row.model')[row.metric]
            self.assertAlmostEqual(row.mean, values.mean())
            self.assertAlmostEqual(row.sd, values.std(ddof=1))
        for _, row in self.tables['test_metrics'].iterrows():
            self.assertAlmostEqual(row['MSE (g²)'], row['RMSE (g)'] ** 2, places=8)

    def test_fold_coefficients_are_converted_to_common_train_scale(self):
        train_positions, _ = next(m1.cv_splitter().split(self.train))
        full = build_ols_reference().fit(self.train[FEATURES], self.train[TARGET])
        fold = build_ols_reference().fit(self.train.iloc[train_positions][FEATURES], self.train.iloc[train_positions][TARGET])
        full_scale = full.named_steps['preprocess'].named_transformers_['numeric'].named_steps['scaler'].scale_[0]
        fold_scale = fold.named_steps['preprocess'].named_transformers_['numeric'].named_steps['scaler'].scale_[0]
        expected = fold.named_steps['regression'].coef_[0] * full_scale / fold_scale
        observed = self.tables['coefficient_folds'].query("model == 'OLS' and fold == 1 and feature == 'flipper_length_mm'").iloc[0]
        self.assertAlmostEqual(observed.coefficient_common_scale, expected)

    def test_external_ols_reference_rejects_different_row_order(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'reference.csv'
            self.tables['test_predictions'][['row_id', 'actual_g', 'OLS']].iloc[::-1].to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'row IDs/order'):
                analyze(self.train, self.test, ols_reference=path)

    def test_artifacts_have_usable_handoff_tables_figures_and_metadata(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / 'results'
            metadata = write_outputs(self.tables, output)
            self.assertEqual(metadata['train_n'], 273)
            self.assertEqual(metadata['test_n'], 69)
            self.assertEqual(metadata['cv_sd_ddof'], 1)
            expected = {'alpha_cv.csv', 'cv_summary.csv', 'test_metrics.csv', 'test_predictions.csv',
                        'group_errors.csv', 'bootstrap.csv', 'coefficients.csv', 'coefficient_path.csv',
                        'pearson_train.csv', 'vif_train.csv'}
            self.assertEqual({path.name for path in output.glob('*.csv')}, expected)
            self.assertFalse((output / 'details').exists())
            self.assertEqual(len(pd.read_csv(output / 'test_metrics.csv')), 3)
            coefficients = pd.read_csv(output / 'coefficients.csv')
            self.assertTrue({'OLS_fold_SD', 'M1_fold_SD'}.issubset(coefficients.columns))
            self.assertIn('L2_norm', pd.read_csv(output / 'coefficient_path.csv').columns)
            self.assertEqual(metadata['test_row_ids'], self.test.index.tolist())
            self.assertEqual(pd.read_csv(output / 'test_predictions.csv').row_id.tolist(), self.test.index.tolist())
            self.assertEqual({path.name for path in output.glob('*.png')},
                             {'actual-predicted.png', 'residuals.png', 'coefficients.png', 'coefficient-path.png'})
            for path in output.glob('*.png'):
                with Image.open(path) as image:
                    self.assertEqual(image.format, 'PNG')
                    self.assertGreater(image.width, 500)
                    image.verify()
            write_outputs(self.tables, output, details=True)
            self.assertEqual(len(list((output / 'details').glob('*.csv'))), 6)

    def test_widget_only_errors_are_not_reported_as_success(self):
        import nbformat
        from source.check_week2 import notebook_errors
        notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("pass")])
        notebook.metadata['widgets'] = {
            'application/vnd.jupyter.widget-state+json': {
                'state': {'widget_id': {'model_name': 'OutputModel', 'state': {
                    'outputs': [{'output_type': 'error', 'ename': 'ValueError',
                                 'evalue': 'Rendering failed inside Output widget', 'traceback': []}],
                }}},
            },
        }
        errors = notebook_errors(notebook)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]['ename'], 'ValueError')

    def test_coefficient_path_uses_fixed_design_and_matches_selected_model(self):
        path = self.tables['coefficient_path']
        self.assertEqual(set(path.alpha), set(m1.ALPHA_CANDIDATES))
        self.assertEqual(path.groupby('alpha').size().tolist(), [6] * 5)
        selected = path[path.alpha == self.tables['selected_alpha']].set_index('feature')
        coefficients = self.tables['coefficients'].set_index('feature')
        np.testing.assert_allclose(selected.loc[coefficients.index, 'coefficient'], coefficients.M1)
        np.testing.assert_allclose(selected.loc[coefficients.index, 'OLS'], coefficients.OLS)
        norms = self.tables['coefficient_path_norms'].sort_values('alpha')
        self.assertTrue((norms.n_train == len(self.train)).all())
        self.assertTrue((np.diff(norms.L2_norm) <= 1e-8).all())
        self.assertTrue((norms.L2_norm < np.linalg.norm(coefficients.OLS)).all())

    def test_vif_matches_auxiliary_least_squares_with_intercept(self):
        ols = build_ols_reference().fit(self.train[FEATURES], self.train[TARGET])
        preprocess = ols.named_steps['preprocess']
        design = preprocess.transform(self.train[FEATURES])
        names = list(preprocess.get_feature_names_out())
        index = names.index('species_Gentoo')
        others = np.column_stack([np.ones(len(design)), np.delete(design, index, axis=1)])
        target = design[:, index]
        fitted = others @ np.linalg.lstsq(others, target, rcond=None)[0]
        expected = np.sum((target - target.mean()) ** 2) / np.sum((target - fitted) ** 2)
        observed = self.tables['vif_train'].set_index('feature').loc['species_Gentoo']
        self.assertAlmostEqual(observed.VIF, expected, places=8)
        self.assertEqual(set(self.tables['vif_train'].feature), set(names))
        self.assertNotIn('const', self.tables['vif_train'].feature.tolist())
        self.assertTrue((self.tables['vif_train'].n_train == len(self.train)).all())
        pearson = self.tables['pearson_train'].set_index('feature')
        numeric = ['flipper_length_mm', 'bill_length_mm', 'bill_depth_mm']
        np.testing.assert_allclose(pearson.loc[numeric, numeric], np.corrcoef(self.train[numeric].to_numpy().T))

    def test_evaluation_cli_does_not_overwrite_manual_markdown(self):
        import os
        import subprocess
        import sys
        with TemporaryDirectory() as directory:
            report = Path(directory) / 'ridge-m1-week2.md'
            original = b'Manually reviewed report: preserve this text.\n'
            report.write_bytes(original)
            subprocess.run(
                [sys.executable, '-B', '-m', 'source.ridge_evaluation',
                 '--output-dir', str(Path(directory) / 'results')],
                cwd=Path(__file__).resolve().parents[1],
                env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1'),
                capture_output=True, text=True, encoding='utf-8', check=True,
            )
            self.assertTrue(report.read_bytes() == original,
                            'Evaluation overwrote manually-written Markdown')
            self.assertEqual(list((Path(directory) / 'results').glob('*.md')), [])
            import json
            metadata = json.loads((Path(directory) / 'results/metadata.json').read_text(encoding='utf-8'))
            self.assertIsNone(metadata['ols_reference']['external_csv'])
            predictions = pd.read_csv(Path(directory) / 'results/test_predictions.csv')
            self.assertEqual(len(predictions), 69)
            self.assertTrue({'OLS', 'M1'}.issubset(predictions.columns))


    def test_project_checks_export_results_only_when_requested(self):
        from source.check_week2 import run_checks
        with patch('source.check_week2.command', return_value='Command completed.'), redirect_stdout(StringIO()) as output:
            with patch('source.check_week2.pd.DataFrame.to_csv') as export:
                run_checks()
                export.assert_not_called()
            self.assertIn('shared_data_protocol', output.getvalue())
            with TemporaryDirectory() as directory:
                run_checks(Path(directory))
                table = pd.read_csv(Path(directory) / 'check-results.csv')
                self.assertTrue({'check', 'owner', 'status', 'detail'}.issubset(table.columns))
