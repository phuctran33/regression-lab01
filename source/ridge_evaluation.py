"""Reproduce Khiem's week 2 Ridge evaluation; never choose alpha using test."""

from argparse import ArgumentParser
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import bootstrap
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate

from .modeling import CATEGORICAL, FEATURES, TARGET, load_penguins, make_pipeline, split_penguins
from .models import m1
from .ridge_diagnostics import train_collinearity, ridge_coefficient_path, save_coefficient_path

ROOT = Path(__file__).resolve().parents[1]
METRICS = {"MAE (g)": "neg_mean_absolute_error", "MSE (g²)": "neg_mean_squared_error",
           "RMSE (g)": "neg_root_mean_squared_error", "R²": "r2"}
NUMERIC = [feature for feature in FEATURES if feature not in CATEGORICAL]
MISSING = "<missing>"


def build_ols_reference():
    """OLS baseline using the shared features and preprocessing pipeline."""
    return make_pipeline(LinearRegression())


def scores(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return {"MAE (g)": mean_absolute_error(actual, predicted), "MSE (g²)": mse,
            "RMSE (g)": np.sqrt(mse), "R²": r2_score(actual, predicted)}


def validate_pairs(actual, ols, ridge):
    for values in (actual, ols, ridge):
        if not isinstance(values, pd.Series) or not values.index.equals(actual.index):
            raise ValueError("Paired predictions must have identical row IDs and order.")
        if not values.index.is_unique or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError("Predictions must have unique IDs and finite values.")
    if len(actual) < 2:
        raise ValueError("At least two paired observations are required.")


def paired_mae_bootstrap(actual, ols, ridge, *, seed=42, n_resamples=2000):
    """Delta is MAE(M1)-MAE(OLS); resample the same row IDs for both models."""
    validate_pairs(actual, ols, ridge)
    if not isinstance(n_resamples, int) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer.")
    ols_error = (actual - ols).to_numpy()
    ridge_error = (actual - ridge).to_numpy()

    def delta(ols_errors, ridge_errors, axis=-1):
        return np.mean(np.abs(ridge_errors) - np.abs(ols_errors), axis=axis)

    result = bootstrap((ols_error, ridge_error), delta, paired=True, vectorized=True,
                       n_resamples=n_resamples, confidence_level=0.95, method="percentile",
                       rng=np.random.default_rng(seed))
    return pd.DataFrame([{
        "contrast": "MAE(M1)-MAE(OLS)", "delta_g": float(delta(ols_error, ridge_error)),
        "ci_low_g": float(result.confidence_interval.low),
        "ci_high_g": float(result.confidence_interval.high),
        "n": len(actual), "n_resamples": n_resamples, "seed": seed,
    }])


def subgroup_errors(test, predictions):
    rows = []
    for category in ("species", "sex"):
        levels = test[category].fillna(MISSING)
        for level in sorted(levels.unique()):
            group = test.index[levels == level]
            for name in ("OLS", "M1"):
                residual = test.loc[group, TARGET] - predictions.loc[group, name]
                rows.append({"feature": category, "level": level, "model": name,
                             "n": len(group), "MAE (g)": residual.abs().mean(),
                             "Mean residual (g)": residual.mean()})
    return pd.DataFrame(rows)


def cv_tables(train, models, splits):
    fold_rows, estimators = [], {}
    for name, model in models.items():
        result = cross_validate(model, train[FEATURES], train[TARGET], cv=splits,
                                scoring=METRICS, return_estimator=True, error_score="raise")
        estimators[name] = result["estimator"]
        for fold in range(len(splits)):
            row = {"model": name, "fold": fold + 1}
            for metric, scorer in METRICS.items():
                value = result[f"test_{metric}"][fold]
                row[metric] = -value if scorer.startswith("neg_") else value
            fold_rows.append(row)
    folds = pd.DataFrame(fold_rows)
    summary = []
    for name, group in folds.groupby("model", sort=False):
        for metric in METRICS:
            summary.append({"model": name, "metric": metric,
                            "mean": group[metric].mean(), "sd": group[metric].std(ddof=1)})
    return folds, pd.DataFrame(summary), estimators


def coefficient_tables(models, fold_models):
    reference = models["OLS"].named_steps["preprocess"]
    names = list(reference.get_feature_names_out())
    reference_scale = reference.named_transformers_["numeric"].named_steps["scaler"].scale_
    encoder = reference.named_transformers_["categorical"].named_steps["encoder"]
    full = []
    for index, feature in enumerate(names):
        row = {"feature": feature, "unit": "g per train SD" if feature in NUMERIC else "g vs reference"}
        for name, model in models.items():
            if list(model.named_steps["preprocess"].get_feature_names_out()) != names:
                raise ValueError("Coefficient comparison requires identical feature encoding.")
            row[name] = model.named_steps["regression"].coef_[index]
        row["abs_M1_minus_abs_OLS"] = abs(row["M1"]) - abs(row["OLS"])
        full.append(row)
    long = []
    for name, estimators in fold_models.items():
        for fold, model in enumerate(estimators, 1):
            preprocess = model.named_steps["preprocess"]
            fold_encoder = preprocess.named_transformers_["categorical"].named_steps["encoder"]
            if any(not np.array_equal(a, b) for a, b in zip(encoder.categories_, fold_encoder.categories_)):
                raise ValueError("Categorical reference levels changed across folds.")
            fold_names = list(preprocess.get_feature_names_out())
            coefficients = np.asarray(model.named_steps["regression"].coef_).copy()
            fold_scale = preprocess.named_transformers_["numeric"].named_steps["scaler"].scale_
            # Reporting conversion only; never used for fitting or predicting validation rows.
            coefficients[:len(NUMERIC)] *= reference_scale / fold_scale
            for feature, coefficient in zip(fold_names, coefficients, strict=True):
                long.append({"model": name, "fold": fold, "feature": feature,
                             "coefficient_common_scale": coefficient})
    long = pd.DataFrame(long)
    stability = long.groupby(["model", "feature"], sort=False)["coefficient_common_scale"].agg(
        n="size", mean="mean", sd="std").reset_index()
    coding = {feature: str(levels[int(drop)]) for feature, levels, drop in zip(
        [f for f in FEATURES if f in CATEGORICAL], encoder.categories_, encoder.drop_idx_, strict=True)}
    return pd.DataFrame(full), long, stability, coding


def analyze(train, test, *, ols_reference=None):
    if not train.index.is_unique or not test.index.is_unique or not train.index.intersection(test.index).empty:
        raise ValueError("Train/test row IDs must be unique and disjoint.")
    search = m1.build_model().fit(train[FEATURES], train[TARGET])
    selected_alpha = float(search.best_params_["regression__alpha"])
    candidates = []
    for i, params in enumerate(search.cv_results_["params"]):
        values = -np.array([search.cv_results_[f"split{k}_test_score"][i] for k in range(5)])
        candidates.append({"alpha": params["regression__alpha"], "mean_MAE_g": values.mean(),
                           "sd_MAE_g": values.std(ddof=1),
                           "selected": i == search.best_index_})
    ridge = search.best_estimator_
    ols = build_ols_reference().fit(train[FEATURES], train[TARGET])
    models = {"OLS": ols, "M1": ridge}
    pearson, vif = train_collinearity(train, ols)
    coefficient_path, path_norms = ridge_coefficient_path(train, ols)
    splits = list(m1.cv_splitter().split(train))
    folds, summary, estimators = cv_tables(train, models, splits)
    predictions = pd.DataFrame({name: model.predict(test[FEATURES]) for name, model in models.items()}, index=test.index)
    predictions.index.name = "row_id"
    validate_pairs(test[TARGET], predictions["OLS"], predictions["M1"])
    reference_status = {"definition": "make_pipeline(LinearRegression())", "external_csv": None}
    if ols_reference is not None:
        external = pd.read_csv(ols_reference).set_index("row_id")
        if not external.index.equals(test.index):
            raise ValueError("OLS reference CSV row IDs/order differ from the held-out test.")
        np.testing.assert_allclose(external["actual_g"], test[TARGET], rtol=0, atol=1e-8)
        np.testing.assert_allclose(external["OLS"], predictions["OLS"], rtol=0, atol=1e-8)
        reference_status.update(external_csv=str(ols_reference), max_gap_g=float(
            np.max(np.abs(external["OLS"].to_numpy() - predictions["OLS"].to_numpy()))))
    test_metrics = pd.DataFrame([{"model": name, **scores(test[TARGET], predictions[name])} for name in models])
    baseline = make_pipeline(Ridge(alpha=m1.BASELINE_ALPHA)).fit(train[FEATURES], train[TARGET])
    baseline_metrics = pd.DataFrame([{"model": "M1 week 1 baseline", "alpha": m1.BASELINE_ALPHA,
                                     **scores(test[TARGET], baseline.predict(test[FEATURES]))}])
    bootstrap_table = paired_mae_bootstrap(test[TARGET], predictions["OLS"], predictions["M1"])
    coefficients, coefficient_folds, stability, coding = coefficient_tables(models, estimators)
    split_rows = []
    for position, row_id in enumerate(train.index):
        validation_fold = next(i + 1 for i, (_, validation) in enumerate(splits) if position in validation)
        split_rows.append({"row_id": int(row_id), "split": "train", "validation_fold": validation_fold})
    split_rows.extend({"row_id": int(row_id), "split": "test", "validation_fold": None} for row_id in test.index)
    records = test[["species", "sex"]].copy()
    records.index.name = "row_id"
    records["actual_g"] = test[TARGET]
    for name in models:
        records[name] = predictions[name]
        records[f"residual_{name}_g"] = test[TARGET] - predictions[name]
    return {
        "alpha_cv": pd.DataFrame(candidates), "cv_folds": folds, "cv_summary": summary,
        "test_metrics": test_metrics, "baseline_metrics": baseline_metrics, "test_predictions": records.reset_index(),
        "group_errors": subgroup_errors(test, predictions), "bootstrap": bootstrap_table,
        "coefficients": coefficients, "coefficient_folds": coefficient_folds,
        "coefficient_stability": stability, "split_manifest": pd.DataFrame(split_rows),
        "pearson_train": pearson, "vif_train": vif,
        "coefficient_path": coefficient_path, "coefficient_path_norms": path_norms,
        "selected_alpha": selected_alpha, "reference_levels": coding, "ols_reference": reference_status,
        "intercepts": {name: float(model.named_steps["regression"].intercept_) for name, model in models.items()},
    }


def save_figures(tables, output_dir):
    save_coefficient_path(tables["coefficient_path"], tables["selected_alpha"], output_dir / "coefficient-path.png")
    records = tables["test_predictions"]
    colors = {"OLS": "#b45825", "M1": "#236cb0"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True, layout="constrained")
    lo = min(records.actual_g.min(), records[["OLS", "M1"]].min().min()) - 100
    hi = max(records.actual_g.max(), records[["OLS", "M1"]].max().max()) + 100
    for ax, name in zip(axes, colors, strict=True):
        ax.scatter(records.actual_g, records[name], alpha=0.7, s=28, color=colors[name])
        ax.plot([lo, hi], [lo, hi], "--", color="#555555", linewidth=1)
        title = "OLS" if name == "OLS" else f"M1 Ridge (alpha={tables['selected_alpha']:g})"
        ax.set(title=title, xlim=(lo, hi), ylim=(lo, hi), aspect="equal")
        ax.grid(alpha=0.15)
    fig.suptitle(f"Held-out test: actual vs predicted (n={len(records)})")
    fig.supxlabel("Actual body mass (g)")
    axes[0].set_ylabel("Predicted body mass (g)")
    fig.savefig(output_dir / "actual-predicted.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, layout="constrained")
    for ax, name in zip(axes, colors):
        ax.scatter(records[name], records[f"residual_{name}_g"], alpha=0.65, s=25, color=colors[name])
        ax.axhline(0, linestyle="--", color="#555555", linewidth=1)
        ax.set(xlabel="Predicted body mass (g)", title=name)
    axes[0].set_ylabel("Residual: actual - predicted (g)")
    fig.savefig(output_dir / "residuals.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    coefficients = tables["coefficients"]
    positions = np.arange(len(coefficients))
    for offset, name in ((-0.12, "OLS"), (0.12, "M1")):
        axes[0].scatter(coefficients[name], positions + offset, label=name, color=colors[name])
        spread = tables["coefficient_stability"].query("model == @name").set_index("feature").loc[coefficients.feature]
        axes[1].scatter(spread.sd, positions + offset, label=name, color=colors[name])
    for ax in axes:
        ax.set_yticks(positions, coefficients.feature)
        ax.grid(axis="x", alpha=0.2)
        ax.legend()
    axes[0].set(xlabel="Coefficient (g per common transformed unit)", title="Full train fit; intercept excluded")
    axes[1].set(xlabel="Sample SD across 5 folds (g)", title="Fold variability on common train scale")
    fig.savefig(output_dir / "coefficients.png", dpi=150)
    plt.close(fig)


def write_outputs(tables, output_dir, *, details=False):
    """Export handoff tables/figures; optional raw fold tables go under details/."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spread = tables["coefficient_stability"].pivot(index="feature", columns="model", values="sd")
    spread = spread.rename(columns={"OLS": "OLS_fold_SD", "M1": "M1_fold_SD"}).reset_index()
    spread.columns.name = None
    coefficients = tables["coefficients"].merge(spread, on="feature", validate="one_to_one")
    path = tables["coefficient_path"].merge(tables["coefficient_path_norms"], on="alpha", validate="many_to_one")
    test_metrics = pd.concat([
        tables["test_metrics"], tables["baseline_metrics"].drop(columns="alpha"),
    ], ignore_index=True)
    handoff = {
        "alpha_cv": tables["alpha_cv"], "cv_summary": tables["cv_summary"],
        "test_metrics": test_metrics, "test_predictions": tables["test_predictions"],
        "group_errors": tables["group_errors"], "bootstrap": tables["bootstrap"],
        "coefficients": coefficients, "coefficient_path": path,
        "pearson_train": tables["pearson_train"], "vif_train": tables["vif_train"],
    }
    for name, table in handoff.items():
        table.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    if details:
        detail_dir = output_dir / "details"
        detail_dir.mkdir(exist_ok=True)
        for name in ("cv_folds", "baseline_metrics", "coefficient_folds",
                     "coefficient_stability", "coefficient_path_norms", "split_manifest"):
            tables[name].to_csv(detail_dir / f"{name}.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    save_figures(tables, output_dir)
    split = tables["split_manifest"]
    train_rows = split.loc[split.split == "train"]
    test_rows = split.loc[split.split == "test"]
    reference = dict(tables["ols_reference"])
    if reference.get("external_csv"):
        reference["csv_sha256"] = hashlib.sha256(Path(reference["external_csv"]).read_bytes()).hexdigest()
    source_paths = [ROOT / "source/models/m1.py", Path(__file__), ROOT / "source/modeling.py",
                    ROOT / "source/check_week2.py", ROOT / "tests/test_ridge_week2.py", ROOT / "source/ridge_diagnostics.py"]
    metadata = {
        "selected_alpha": tables["selected_alpha"], "baseline_alpha": m1.BASELINE_ALPHA,
        "alpha_candidates": list(m1.ALPHA_CANDIDATES),
        "train_n": len(train_rows), "test_n": len(test_rows), "features": FEATURES, "target": TARGET,
        "seed": 42, "cv_folds": 5, "cv_sd_ddof": 1, "cv_status": "development / reused for tuning",
        "train_row_ids": train_rows.row_id.astype(int).tolist(),
        "test_row_ids": test_rows.row_id.astype(int).tolist(),
        "validation_row_ids": {str(fold): train_rows.loc[train_rows.validation_fold == fold, "row_id"].astype(int).tolist()
                               for fold in range(1, 6)},
        "residual_definition": "actual - predicted", "bootstrap": "paired, 2000, percentile 95%, seed 42",
        "reference_levels": tables["reference_levels"], "intercepts_g": tables["intercepts"],
        "ols_reference": reference, "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib")},
        "dataset_sha256": hashlib.sha256((ROOT / "data/penguins.csv").read_bytes()).hexdigest(),
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
    }
    try:
        metadata["base_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        metadata["base_commit"] = None
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/m1-week2")
    parser.add_argument("--ols-reference", type=Path, help="Optional CSV for cross-checking independently computed OLS predictions.")
    parser.add_argument("--details", action="store_true", help="Also export raw fold/intermediate tables under details/.")
    args = parser.parse_args()
    train, test = split_penguins(load_penguins())
    tables = analyze(train, test, ols_reference=args.ols_reference)
    write_outputs(tables, args.output_dir, details=args.details)
    print("Selected alpha:", tables["selected_alpha"])
    print(tables["test_metrics"].to_string(index=False))
    print(tables["bootstrap"].to_string(index=False))
    print("Outputs: 10 CSV tables, 4 PNG figures, metadata.json")
    if args.details:
        print("Raw fold/intermediate tables:", args.output_dir / "details")
    print("Output directory:", args.output_dir.resolve())


if __name__ == "__main__":
    main()
