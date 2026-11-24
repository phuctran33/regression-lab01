"""Reproduce the OLS and M3 tables and figures for report section 3.3."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import bootstrap
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_validate
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera

from .modeling import CATEGORICAL, FEATURES, TARGET, load_penguins, regression_metrics, split_penguins
from .models import m3, ols

NUMERIC = [feature for feature in FEATURES if feature not in CATEGORICAL]
LABELS = {
    "flipper_length_mm": "Chiều dài cánh bơi (mm)",
    "bill_length_mm": "Chiều dài mỏ (mm)",
    "bill_depth_mm": "Độ sâu mỏ (mm)",
}
MODELS = (ols, m3)
COLORS = {"OLS": "#a54c20", "M3": "#156581"}
SEED = 42
FIGURES = Path(__file__).resolve().parents[1] / "report" / "figures"


def ols_design(pipeline, rows):
    """Use the fitted sklearn design and retain its column order and row labels."""
    preprocess = pipeline.named_steps["preprocess"]
    design = pd.DataFrame(
        preprocess.transform(rows[FEATURES]),
        index=rows.index,
        columns=preprocess.get_feature_names_out(),
    )
    return sm.add_constant(design, has_constant="add")


def fit_ols_inference(pipeline, train):
    result = sm.OLS(train[TARGET], ols_design(pipeline, train), missing="raise").fit()
    expected = pipeline.predict(train[FEATURES])
    np.testing.assert_allclose(result.fittedvalues, expected, rtol=0, atol=1e-8)
    return result, float(np.abs(result.fittedvalues - expected).max())


def ols_prediction_intervals(pipeline, result, rows):
    """Return the classical OLS 95% mean and individual intervals for raw inputs."""
    return result.get_prediction(ols_design(pipeline, rows)).summary_frame(alpha=0.05)


def ols_tables(result, train):
    design = result.model.exog
    terms = result.model.exog_names
    robust = result.get_robustcov_results(cov_type="HC3", use_t=True)
    coefficient_selectors = np.eye(len(terms))
    joint_hc3 = robust.f_test(coefficient_selectors[1:])
    species_restrictions = coefficient_selectors[[term.startswith("species_") for term in terms]]
    species_tests = pd.DataFrame.from_dict({
        label: (float(test.fvalue), int(test.df_num), int(test.df_denom), float(test.pvalue))
        for label, test in (
            ("Partial F (classical)", result.f_test(species_restrictions)),
            ("Wald F (HC3)", robust.f_test(species_restrictions)),
        )
    }, orient="index", columns=["F", "df numerator", "df denominator", "p"])
    classical_ci = result.conf_int().to_numpy()
    robust_ci = robust.conf_int()
    coefficients = pd.DataFrame({
        "Coefficient": result.params,
        "SE": result.bse,
        "t": result.tvalues,
        "p": result.pvalues,
        "CI low": classical_ci[:, 0],
        "CI high": classical_ci[:, 1],
        "SE HC3": robust.bse,
        "t HC3": robust.tvalues,
        "p HC3": robust.pvalues,
        "CI HC3 low": robust_ci[:, 0],
        "CI HC3 high": robust_ci[:, 1],
    })
    jb, jb_p, skew, kurtosis = jarque_bera(result.resid)
    bp, bp_p, _, _ = het_breuschpagan(result.resid, design, robust=True)
    reset = linear_reset(result, power=3, use_f=True, cov_type="HC3")
    tests = pd.DataFrame.from_dict({
        "RESET F (HC3)": (
            float(reset.fvalue), f"{int(reset.df_num)}, {int(reset.df_denom)}", float(reset.pvalue),
        ),
        "Jarque–Bera χ²": (jb, "2", jb_p),
        "Breusch–Pagan/Koenker LM χ²": (bp, str(int(result.df_model)), bp_p),
    }, orient="index", columns=["Statistic", "df", "p"])
    vif = pd.Series(
        [variance_inflation_factor(design, j) for j in range(1, len(terms))],
        index=terms[1:], name="VIF",
    )
    influence = result.get_influence()
    observations = train[["species", "sex", TARGET]].assign(
        fitted=result.fittedvalues, residual=result.resid,
        studentized=influence.resid_studentized_external,
        leverage=influence.hat_matrix_diag, cook=influence.cooks_distance[0],
    )
    n, p = design.shape
    fit = pd.Series({
        "n": n, "R²": result.rsquared, "Adjusted R²": result.rsquared_adj,
        "df model": result.df_model, "df residual": result.df_resid,
        "SSE (g²)": result.ssr, "Residual variance (g²)": result.mse_resid,
        "Residual standard error (g)": np.sqrt(result.mse_resid),
        "F": result.fvalue, "p F": result.f_pvalue,
        "F HC3": float(joint_hc3.fvalue), "p F HC3": float(joint_hc3.pvalue),
        "Skewness": skew, "Kurtosis": kurtosis,
        "Count |studentized| > 3": (observations.studentized.abs() > 3).sum(),
        "Count leverage > 2p/n": (observations.leverage > 2 * p / n).sum(),
        "Count Cook > 4/n": (observations.cook > 4 / n).sum(),
    })
    return {
        "OLS train fit": fit,
        "OLS coefficients": coefficients,
        "OLS joint species tests": species_tests,
        "OLS train diagnostics": tests,
        "Train numeric Pearson": train[NUMERIC].corr(),
        "Train VIF": vif,
        "Train largest Cook distances": observations.nlargest(5, "cook").rename_axis("CSV index"),
    }


def train_cv(train):
    scores = {
        "MAE (g)": "neg_mean_absolute_error",
        "MSE (g²)": "neg_mean_squared_error",
        "RMSE (g)": "neg_root_mean_squared_error",
        "R²": "r2",
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for module in MODELS:
        result = cross_validate(
            module.build_model(), train[FEATURES], train[TARGET],
            cv=cv, scoring=scores, error_score="raise",
        )
        for metric, scorer in scores.items():
            values = result[f"test_{metric}"]
            if scorer.startswith("neg_"):
                values = -values
            rows.append((module.NAME, metric, values.mean(), values.std(ddof=1)))
    return pd.DataFrame(rows, columns=["Model", "Metric", "Mean", "SD"]).set_index(["Model", "Metric"])


def test_predictions(models, test):
    return pd.DataFrame({
        name: model.predict(test[FEATURES]) for name, model in models.items()
    }, index=test.index)


def heldout_tables(test, predictions):
    actual = test[TARGET]
    if not predictions.index.equals(actual.index):
        raise ValueError("Test predictions must have the same index and row order as test observations.")
    residuals = predictions.rsub(actual, axis=0)
    scores = pd.DataFrame.from_dict({
        name: {
            **regression_metrics(actual, predicted),
            "MSE (g²)": mean_squared_error(actual, predicted),
        }
        for name, predicted in predictions.items()
    }, orient="index")[["MAE (g)", "MSE (g²)", "RMSE (g)", "R²"]]
    groups = []
    for category in ("species", "sex"):
        for level, group in residuals.groupby(test[category].fillna("<thiếu>")):
            for name, values in group.items():
                groups.append((
                    category, level, name, len(values),
                    values.abs().mean(), np.sqrt(values.pow(2).mean()), values.mean(),
                ))
    subgroup = pd.DataFrame(groups, columns=[
        "Feature", "Level", "Model", "n", "MAE (g)", "RMSE (g)", "Mean residual (g)",
    ]).set_index(["Feature", "Level", "Model"])
    return {
        "Held-out test": scores,
        "Test residuals actual − predicted": residuals.describe().T,
        "Test errors by species or sex": subgroup,
        "Paired test bootstrap": paired_bootstrap(residuals),
    }


def paired_bootstrap(residuals, *, seed=SEED, n_resamples=2000):
    def deltas(ols_errors, m3_errors, axis=-1):
        delta_mae = np.mean(np.abs(m3_errors) - np.abs(ols_errors), axis=axis)
        delta_rmse = np.sqrt(np.mean(m3_errors ** 2, axis=axis)) - np.sqrt(
            np.mean(ols_errors ** 2, axis=axis)
        )
        return np.stack((delta_mae, delta_rmse))

    pairs = (residuals[ols.NAME].to_numpy(), residuals[m3.NAME].to_numpy())
    result = bootstrap(
        pairs, deltas, paired=True, n_resamples=n_resamples, method="percentile",
        confidence_level=0.95, rng=np.random.default_rng(seed),
    )
    return pd.DataFrame({
        "Delta (g)": deltas(*pairs),
        "Lower 95% (g)": result.confidence_interval.low,
        "Upper 95% (g)": result.confidence_interval.high,
    }, index=["MAE", "RMSE"])


def save_ols_diagnostics(result, train, path):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    covariates = [("Dự đoán (g)", result.fittedvalues)]
    covariates.extend((LABELS.get(feature, feature), train[feature]) for feature in NUMERIC)
    for ax, (label, x) in zip(axes.flat, covariates):
        ax.scatter(x, result.resid, s=13, alpha=0.45, color=COLORS["M3"])
        trend = sm.nonparametric.lowess(result.resid, x, frac=0.6)
        ax.plot(*trend.T, color=COLORS["OLS"], lw=1.8, label="LOWESS")
        ax.axhline(0, color="#49566c", ls="--", lw=1)
        ax.set(xlabel=label, ylabel="Phần dư (g)")
    axes.flat[0].legend(frameon=False)
    axes.flat[4].hist(result.resid, bins=16, color=COLORS["M3"], edgecolor="white")
    axes.flat[4].set(xlabel="Phần dư (g)", ylabel="Số mẫu", title="Histogram")
    sm.qqplot(result.resid, line="s", ax=axes.flat[5], markerfacecolor=COLORS["M3"], alpha=0.55)
    axes.flat[5].set(xlabel="Phân vị chuẩn lý thuyết", ylabel="Phân vị phần dư (g)", title="Q–Q")
    fig.suptitle(f"Chẩn đoán OLS trên train (n = {len(train)})")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_test_comparison(actual, predictions, path):
    fig, axes = plt.subplot_mosaic(
        [["OLS observed", "M3 observed", "residuals"], ["OLS fitted", "M3 fitted", "residuals"]],
        figsize=(12, 7), layout="constrained",
    )
    residuals = predictions.rsub(actual, axis=0)
    limits = (min(actual.min(), predictions.min().min()), max(actual.max(), predictions.max().max()))
    residual_extent = residuals.abs().max().max() * 1.05
    bins = np.histogram_bin_edges(residuals.to_numpy().ravel(), bins=14)
    for name, predicted in predictions.items():
        ax = axes[f"{name} observed"]
        ax.scatter(actual, predicted, s=22, alpha=0.6, color=COLORS[name])
        ax.plot(limits, limits, ls="--", lw=1, color="#49566c")
        ax.set(xlabel="Thực tế (g)", ylabel="Dự đoán (g)", title=name, aspect="equal")
        residual_ax = axes[f"{name} fitted"]
        residual_ax.scatter(predicted, residuals[name], s=22, alpha=0.6, color=COLORS[name])
        residual_ax.axhline(0, color="#49566c", ls="--", lw=1)
        residual_ax.set(
            xlabel="Dự đoán (g)", ylabel="Phần dư (g)", xlim=limits,
            ylim=(-residual_extent, residual_extent),
        )
        axes["residuals"].hist(
            residuals[name], bins=bins, histtype="step", linewidth=2, color=COLORS[name],
            linestyle="--" if name == "OLS" else "-", label=name,
        )
    histogram = axes["residuals"]
    histogram.axvline(0, color="#49566c", ls=":", lw=1)
    histogram.set(xlabel="Thực tế − dự đoán (g)", ylabel="Số mẫu", title="Phân phối phần dư")
    histogram.legend(frameon=False)
    fig.suptitle(f"Dự đoán trên test (n = {len(actual)})")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_smooth_effects(models, train, path):
    pipeline = models[m3.NAME]
    spline_model = pipeline.named_steps["regression"]
    preprocess = pipeline.named_steps["preprocess"]
    train_design = preprocess.transform(train[FEATURES])
    scale = preprocess.named_transformers_["numeric"].named_steps["scaler"]
    slopes = models[ols.NAME].named_steps["regression"].coef_
    fig, axes = plt.subplots(2, len(spline_model.smooth_indices_), figsize=(12, 6),
                             sharex="col", sharey="row", squeeze=False, layout="constrained")
    curvature = []
    terms = zip(spline_model.smooth_indices_, spline_model.splines_, spline_model.centers_, strict=True)
    for column, (index, spline, center) in enumerate(terms):
        feature = NUMERIC[index]
        observed = train_design[:, index]
        observed_effect = spline(observed) - center
        affine = np.polyfit(observed, observed_effect, 1)
        grid = np.linspace(observed.min(), observed.max(), 250)
        raw = grid * scale.scale_[index] + scale.mean_[index]
        effect = spline(grid) - center
        ols_effect = slopes[index] * grid
        departure = effect - np.polyval(affine, grid)
        curvature.append((
            feature, np.sqrt(np.mean((observed_effect - np.polyval(affine, observed)) ** 2)),
            np.abs(departure).max(), np.abs(effect - ols_effect).max(),
        ))
        top, bottom = axes[:, column]
        top.plot(raw, effect, color=COLORS["M3"], lw=2, label="M3")
        top.plot(raw, ols_effect, color=COLORS["OLS"], ls="--", lw=1.8, label="OLS")
        top.set_title(LABELS.get(feature, feature))
        bottom.plot(raw, departure, color=COLORS["M3"], lw=2)
        bottom.axhline(0, color="#49566c", ls="--", lw=1)
        bottom.set_xlabel(LABELS.get(feature, feature))
        for ax in (top, bottom):
            ax.grid(axis="y", alpha=0.2)
    axes[0, 0].set_ylabel("Đóng góp đã định tâm (g)")
    axes[1, 0].set_ylabel("M3 − đường thẳng riêng (g)")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Hiệu ứng theo từng số đo trên khoảng train")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return pd.DataFrame(curvature, columns=[
        "Feature", "RMS departure at training observations (g)", "Max departure on grid (g)",
        "Max difference from OLS on grid (g)",
    ]).set_index("Feature")


def analyze(train, test, figure_dir):
    models = {
        module.NAME: module.build_model().fit(train[FEATURES], train[TARGET]) for module in MODELS
    }
    inference, gap = fit_ols_inference(models[ols.NAME], train)
    tables = ols_tables(inference, train)
    tables["OLS prediction parity, max gap (g)"] = gap
    # Measurements of Adelie female at row index 130 in the shared training split.
    example = pd.DataFrame([[190.0, 38.5, 17.9, "Adelie", "female"]], columns=FEATURES)
    tables["OLS prediction example, input"] = example
    tables["OLS prediction example, 95% intervals (g)"] = ols_prediction_intervals(
        models[ols.NAME], inference, example,
    )
    tables["Train CV, 5 folds"] = train_cv(train)
    spline = models[m3.NAME].named_steps["regression"]
    tables["M3 train configuration"] = pd.Series({
        "lambda": spline.lam, "sweeps": spline.n_iter_, "max sweeps": spline.max_iter,
        "tolerance (g)": spline.tol, "last max change (g)": spline.max_change_,
        "converged": spline.converged_,
    })
    figure_dir.mkdir(parents=True, exist_ok=True)
    save_ols_diagnostics(inference, train, figure_dir / "section-3-3-ols-diagnostics.png")
    tables["M3 empirical curvature"] = save_smooth_effects(
        models, train, figure_dir / "section-3-3-smooth-effects.png"
    )
    predictions = test_predictions(models, test)
    tables.update(heldout_tables(test, predictions))
    save_test_comparison(test[TARGET], predictions, figure_dir / "section-3-3-test-predictions.png")
    return tables


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", type=Path, default=FIGURES)
    args = parser.parse_args()
    train, test = split_penguins(load_penguins())
    for name, table in analyze(train, test, args.figure_dir).items():
        print(f"\n{name}")
        if isinstance(table, (pd.DataFrame, pd.Series)):
            print(table.to_string(float_format=lambda value: f"{value:.7g}"))
        else:
            print(table)


if __name__ == "__main__":
    main()
