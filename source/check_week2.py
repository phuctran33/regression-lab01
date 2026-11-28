"""Smoke-check the week 2 handoff; record missing work rather than passing placeholders."""

from argparse import ArgumentParser
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from .modeling import FEATURES, TARGET, load_penguins, split_penguins

ROOT = Path(__file__).resolve().parents[1]


def command(args, *, cwd=ROOT, input_text=None, timeout=240):
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg')
    result = subprocess.run([sys.executable, '-B', *args], cwd=cwd, env=env, input=input_text,
                            capture_output=True, text=True, encoding='utf-8', timeout=timeout)
    if result.returncode:
        raise RuntimeError((result.stdout + '\n' + result.stderr)[-4500:])
    return result.stdout + result.stderr


def notebook_errors(notebook):
    """Include widget-held exceptions: Output widgets can suppress cell errors."""
    failures = [out for cell in notebook.cells for out in cell.get('outputs', [])
                if out.get('output_type') == 'error']

    def inspect(value):
        if isinstance(value, dict):
            if value.get('output_type') == 'error':
                failures.append(value)
            for child in value.values():
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)

    inspect(notebook.metadata.get('widgets', {}))
    return failures

def execute_notebook():
    import nbformat
    from nbclient import NotebookClient
    temporary = Path(tempfile.mkdtemp(prefix='lab01-week2-notebook-'))
    for folder in ('source', 'data'):
        shutil.copytree(ROOT / folder, temporary / folder,
                        ignore=shutil.ignore_patterns('__pycache__', '.ipynb_checkpoints'))
    kernel_name = 'lab01-week2-check'
    kernel_dir = temporary / 'jupyter/kernels' / kernel_name
    kernel_dir.mkdir(parents=True)
    (kernel_dir / 'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'Lab01 week2 check', 'language': 'python',
    }), encoding='utf-8')
    for name, folder in (('JUPYTER_PATH', 'jupyter'), ('IPYTHONDIR', 'ipython'), ('JUPYTER_RUNTIME_DIR', 'runtime')):
        os.environ[name] = str(temporary / folder)
    notebook = nbformat.read(temporary / 'source/lab01.ipynb', as_version=4)
    notebook.cells.append(nbformat.v4.new_code_cell('''
import json
for name, pipeline in models.items():
    prediction = pipeline.predict(train_data[FEATURES].iloc[:1])
    assert np.isfinite(prediction).all(), name
checked_app = globals().get("prediction_app", globals().get("app"))
if checked_app is not None and hasattr(checked_app, "predict_button"):
    for feature in FEATURES:
        checked_app.controls[feature].value = train_data[feature].dropna().iloc[0]
    # Check the benchmark using its full-train predictor, then the separate calibrated fit.
    if hasattr(checked_app, "protocol"):
        checked_app.protocol.value = "full"
    for name in models:
        checked_app.model_picker.value = name
        checked_app.predict_button.click()
        assert name in checked_app.result.value and "Không thể" not in checked_app.result.value
        assert bytes(checked_app.chart.value).startswith(b"\\x89PNG")
        inputs = {feature: control.value for feature, control in checked_app.controls.items()}
        expected = models[name].predict(pd.DataFrame([inputs]))[0]
        assert f"{expected:,.1f}" in checked_app.result.value
    if getattr(checked_app, "calibrated", {}):
        checked_app.protocol.value = "conformal"
        for name, fitted in checked_app.calibrated.items():
            checked_app.model_picker.value = name
            checked_app.predict_button.click()
            point, low, high = fitted.interval(pd.DataFrame([inputs]), checked_app.level.value)
            assert all(f"{value[0]:,.1f}" in checked_app.result.value for value in (point, low, high))
elif "render_prediction" in globals():
    import builtins
    from io import BytesIO
    from unittest.mock import patch
    printed, charts = [], []
    real_print, real_show = builtins.print, plt.show
    def capture_print(*args, **kwargs):
        printed.append(" ".join(str(arg) for arg in args))
        return real_print(*args, **kwargs)
    def capture_show(*args, **kwargs):
        image = BytesIO()
        plt.gcf().savefig(image, format="png")
        charts.append(image.getvalue())
        return real_show(*args, **kwargs)
    for feature in FEATURES:
        controls[feature].value = train_data[feature].dropna().iloc[0]
    with patch("builtins.print", capture_print), patch.object(plt, "show", capture_show):
        for name in models:
            model_picker.value = name
            printed.clear()
            charts.clear()
            predict_button.click()
            inputs = {feature: control.value for feature, control in controls.items()}
            expected = models[name].predict(pd.DataFrame([inputs]))[0]
            assert any(f"{expected:,.0f}" in line for line in printed), name
            assert charts and charts[-1].startswith(b"\\x89PNG"), name
else:
    raise AssertionError("No prediction application was found")
print("CHECK_MODELS=" + json.dumps(list(models)))
'''))
    NotebookClient(notebook, timeout=240, kernel_name=kernel_name,
                   resources={'metadata': {'path': str(temporary / 'source')}}).execute()
    outputs = [out for cell in notebook.cells for out in cell.get('outputs', [])]
    failures = notebook_errors(notebook)
    if failures:
        raise RuntimeError(str(failures))
    names = None
    for output in outputs:
        if output.get('output_type') == 'stream':
            for line in output.get('text', '').splitlines():
                if line.startswith('CHECK_MODELS='):
                    names = json.loads(line.partition('=')[2])
    nbformat.write(notebook, temporary / 'executed.ipynb')
    return names, temporary / 'executed.ipynb'


def run_checks(output_dir=None, *, notebook=False):
    rows = []

    def check(name, owner, operation):
        try:
            detail = operation()
            rows.append({'check': name, 'owner': owner, 'status': 'PASS', 'detail': str(detail)})
        except Exception as error:
            rows.append({'check': name, 'owner': owner, 'status': 'FAIL',
                         'detail': f'{type(error).__name__}: {error}'})

    def data_protocol():
        train, test = split_penguins(load_penguins())
        assert len(train) == 273 and len(test) == 69
        assert train.index.intersection(test.index).empty
        assert FEATURES == ['flipper_length_mm', 'bill_length_mm', 'bill_depth_mm', 'species', 'sex']
        return '5 shared inputs; 273 train / 69 test; seed 42'

    def requirements():
        for line in (ROOT / 'requirements.txt').read_text(encoding='utf-8').splitlines():
            if '==' in line and not line.strip().startswith('#'):
                name, version = line.strip().split('==')
                actual = importlib.metadata.version(name)
                if actual != version:
                    raise ValueError(f'{name}: expected {version}, installed {actual}')
        return 'All pinned requirements match installed versions.'

    def ready(name):
        module = importlib.import_module('source.models.' + name)
        if getattr(module, 'ESTIMATOR', None) is None:
            raise ValueError(f'{name.upper()} is still ESTIMATOR=None; CLI runs the example OLS.')
        train, test = split_penguins(load_penguins())
        model = module.build_model().fit(train[FEATURES], train[TARGET])
        assert np.isfinite(model.predict(test[FEATURES])).all()
        return f'{type(model.named_steps["regression"]).__name__}; 69 finite test predictions'

    check('shared_data_protocol', 'Nhóm', data_protocol)
    check('requirements_versions', 'Khiêm', requirements)
    for name, owner in (('m1', 'Khiêm'), ('m2', 'Hy'), ('m3', 'Hưng'), ('m4', 'Phúc')):
        check(f'{name}_cli', owner, lambda name=name: command(['-m', 'source.models.' + name]).strip())
        check(f'{name}_configured', owner, lambda name=name: ready(name))
    check('unit_suite', 'Khiêm / chủ module', lambda: command(['-m', 'unittest', 'discover', '-s', 'tests', '-v']).strip() or 'Unittest suite passed; details on stderr.')

    def compare_ols():
        if not (ROOT / 'source/models/ols.py').exists():
            raise ValueError('Shared OLS module is not integrated on this branch yet.')
        from .ridge_evaluation import build_ols_reference
        module = importlib.import_module('source.models.ols')
        train, test = split_penguins(load_penguins())
        shared = module.build_model().fit(train[FEATURES], train[TARGET])
        reference = build_ols_reference().fit(train[FEATURES], train[TARGET])
        np.testing.assert_allclose(shared.predict(test[FEATURES]), reference.predict(test[FEATURES]),
                                   rtol=0, atol=1e-8)
        return 'Shared OLS matches the M1 comparison baseline on all 69 test rows.'

    check('shared_ols_baseline', 'Hưng / Khiêm', compare_ols)
    if notebook:
        state = {}

        def notebook_check():
            names, path = execute_notebook()
            state['models'] = names or []
            return f'Executed notebook and M1 prediction widget; output={path}'

        check('notebook_and_m1_app', 'Hưng / Khiêm', notebook_check)

        def full_app():
            names = set(state.get('models', []))
            missing = {'OLS', 'M1', 'M2', 'M3', 'M4'} - names
            if missing:
                raise ValueError('Missing app choices: ' + ', '.join(sorted(missing)))
            # A model name backed by a placeholder does not count as completion.
            for name in ('m1', 'm2', 'm3', 'm4'):
                if getattr(importlib.import_module('source.models.' + name), 'ESTIMATOR', None) is None:
                    raise ValueError(f'{name.upper()} choice still uses a placeholder.')
            return 'All 5 real models available.'

        check('five_real_models_in_app', 'Hưng / Phúc', full_app)
    table = pd.DataFrame(rows)
    for row in rows:
        print(f"{row['status']:4} | {row['check']} | {row['owner']}")
        print('    ' + row['detail'])
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / 'check-results.csv'
        table.to_csv(path, index=False, encoding='utf-8-sig')
        print('Check details:', path.resolve())
    return int((table.status == 'FAIL').any())


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, help='Optionally save check-results.csv; otherwise print results only.')
    parser.add_argument('--notebook', action='store_true')
    args = parser.parse_args()
    return run_checks(args.output_dir, notebook=args.notebook)


if __name__ == '__main__':
    raise SystemExit(main())
