"""Reproduce the M4 tuning, held-out evaluation and train sensitivity analysis."""

from argparse import ArgumentParser
from pathlib import Path
from warnings import catch_warnings, simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import bootstrap
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, KFold, cross_validate

from .eda_statistics import distribution_audit
from .modeling import FEATURES, TARGET, load_penguins, make_pipeline, regression_metrics, split_penguins
from .models import m4

SEED = 42
BASELINE = "OLS"
MODEL_NAMES = (BASELINE, m4.NAME)
GRID = {
    "regression__epsilon": (1.1, 1.2, 1.35, 1.5, 1.75, 2.0, 2.5, 3.0),
    "regression__alpha": (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
}
SCORING = {
    "MAE (g)": "neg_mean_absolute_error",
    "MSE (g²)": "neg_mean_squared_error",
    "RMSE (g)": "neg_root_mean_squared_error",
    "R²": "r2",
}


def tune_huber(train):
    """Rank candidate parameters by MAE on the shared train folds."""
    search = GridSearchCV(
        m4.build_model(), GRID, scoring=SCORING["MAE (g)"],
        cv=KFold(n_splits=5, shuffle=True, random_state=SEED), refit=False, error_score="raise",
    )
    with catch_warnings():
        simplefilter("error", ConvergenceWarning)
        search.fit(train[FEATURES], train[TARGET])
    scores = search.cv_results_
    fold_scores = np.column_stack([
        -scores[f"split{fold}_test_score"] for fold in range(search.n_splits_)
    ])
    return pd.DataFrame({
        "epsilon": scores["param_regression__epsilon"],
        "alpha": scores["param_regression__alpha"],
        "MAE TB (g)": fold_scores.mean(axis=1),
        "MAE SD (g)": fold_scores.std(axis=1, ddof=1),
    }).sort_values("MAE TB (g)", ignore_index=True)


def _cv_summary(models, train):
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for name, model in models.items():
        scores = cross_validate(
            model, train[FEATURES], train[TARGET], cv=cv, scoring=SCORING, error_score="raise",
        )
        for metric, scorer in SCORING.items():
            values = scores[f"test_{metric}"]
            if scorer.startswith("neg_"):
                values = -values
            rows.append((name, metric, values.mean(), values.std(ddof=1)))
    return pd.DataFrame(rows, columns=["Mô hình", "Chỉ số", "TB", "SD"]).set_index(["Mô hình", "Chỉ số"])


def _predict(models, data):
    return data[["species", "sex", TARGET]].assign(
        **{name: model.predict(data[FEATURES]) for name, model in models.items()},
    )


def paired_mae_interval(predictions):
    """Resample the paired difference in absolute errors for each test row."""
    actual = predictions[TARGET]
    delta = (actual - predictions[m4.NAME]).abs() - (actual - predictions[BASELINE]).abs()
    result = bootstrap(
        (delta.to_numpy(),), np.mean, n_resamples=2000, method="percentile",
        confidence_level=0.95, rng=np.random.default_rng(SEED),
    )
    return pd.Series({
        "n": len(delta),
        "ΔMAE M4 − OLS (g)": delta.mean(),
        "CI 95% dưới (g)": result.confidence_interval.low,
        "CI 95% trên (g)": result.confidence_interval.high,
    })


def _subgroup_errors(predictions):
    rows = []
    for feature in ("species", "sex"):
        for level, group in predictions.groupby(predictions[feature].fillna("<thiếu>")):
            for name in MODEL_NAMES:
                residual = group[TARGET] - group[name]
                rows.append((
                    feature, level, name, len(group),
                    residual.abs().mean(), np.sqrt(residual.pow(2).mean()), residual.mean(),
                ))
    return pd.DataFrame(rows, columns=[
        "Biến", "Nhóm", "Mô hình", "n", "MAE (g)", "RMSE (g)", "Phần dư TB (g)",
    ]).set_index(["Biến", "Nhóm", "Mô hình"])


def _train_influence(models, train, predictions):
    preprocessor = models[BASELINE].named_steps["preprocess"]
    design = sm.add_constant(preprocessor.transform(train[FEATURES]), has_constant="add")
    fit = sm.OLS(train[TARGET], design, missing="raise").fit()
    np.testing.assert_allclose(fit.fittedvalues, predictions[BASELINE], rtol=0, atol=1e-8)
    influence = fit.get_influence()
    huber = models[m4.NAME].named_steps["regression"]
    _, eda_flags = distribution_audit(train)
    mass_flags = eda_flags.loc[eda_flags["Biến"].eq(TARGET), "Chỉ số CSV"]
    cases = predictions.assign(
        **{
            "Phần dư OLS (g)": train[TARGET] - predictions[BASELINE],
            "Phần dư M4 (g)": train[TARGET] - predictions[m4.NAME],
            "Studentized OLS": influence.resid_studentized_external,
            "Leverage OLS": influence.hat_matrix_diag,
            "Cook OLS": influence.cooks_distance[0],
            "Cờ IQR trong EDA": train.index.isin(eda_flags["Chỉ số CSV"]),
            "Cờ IQR khối lượng": train.index.isin(mass_flags),
            "Vùng tuyến tính Huber": huber.outliers_,
        },
    ).rename_axis("Chỉ số CSV")
    n, p = design.shape
    summary = pd.Series({
        "Train n": n,
        "|Studentized| > 3": cases["Studentized OLS"].abs().gt(3).sum(),
        "Leverage > 2p/n": cases["Leverage OLS"].gt(2 * p / n).sum(),
        "Cook > 4/n": cases["Cook OLS"].gt(4 / n).sum(),
        "Số vòng lặp Huber": huber.n_iter_,
        "Số điểm vùng tuyến tính Huber": huber.outliers_.sum(),
        "Ngưỡng epsilon × scale (g)": huber.epsilon * huber.scale_,
        "Số điểm vừa có cờ IQR vừa thuộc vùng tuyến tính": (
            cases["Cờ IQR trong EDA"] & cases["Vùng tuyến tính Huber"]
        ).sum(),
    })
    return cases, summary


def _sensitivity(models, train, cases):
    by_cook = cases["Cook OLS"].nlargest(3).index
    by_residual = cases["Studentized OLS"].abs().nlargest(3).index
    scenarios = {
        "Bỏ điểm có |studentized| cao nhất": by_residual[:1],
        "Bỏ 3 điểm có Cook cao nhất": by_cook,
    }
    rows = []
    for scenario, removed in scenarios.items():
        retained = train.drop(index=removed)
        for name, model in models.items():
            refit = clone(model).fit(retained[FEATURES], retained[TARGET])
            delta = (refit.predict(retained[FEATURES]) - cases.loc[retained.index, name]).abs()
            rows.append((
                scenario, name, ", ".join(map(str, removed)), len(retained), delta.mean(), delta.max(),
            ))
    changes = pd.DataFrame(rows, columns=[
        "Kịch bản", "Mô hình", "CSV bỏ khi fit", "n còn lại", "|Δ dự đoán| TB (g)", "|Δ dự đoán| max (g)",
    ]).set_index(["Kịch bản", "Mô hình"])
    flagged_mass = cases.index[cases["Cờ IQR khối lượng"]]
    return changes, cases.loc[by_cook.union(by_residual, sort=False).union(flagged_mass, sort=False)]


def evaluate_huber(train, test):
    grid = tune_huber(train)
    models = {BASELINE: make_pipeline(), m4.NAME: m4.build_model()}
    with catch_warnings():
        simplefilter("error", ConvergenceWarning)
        cv_summary = _cv_summary(models, train)
        for model in models.values():
            model.fit(train[FEATURES], train[TARGET])
        train_cases, influence_summary = _train_influence(models, train, _predict(models, train))
        sensitivity, cases = _sensitivity(models, train, train_cases)
    held_out = _predict(models, test)

    fits = [model.named_steps["regression"] for model in models.values()]
    features = models[BASELINE].named_steps["preprocess"].get_feature_names_out()
    coefficients = pd.DataFrame(
        {name: np.r_[fit.intercept_, fit.coef_] for name, fit in zip(models, fits)},
        index=["Hệ số chặn", *features],
    )
    coefficients["M4 − OLS (g)"] = coefficients[m4.NAME] - coefficients[BASELINE]
    metrics = pd.DataFrame.from_dict({
        name: regression_metrics(test[TARGET], held_out[name]) for name in models
    }, orient="index")
    residuals = held_out[list(MODEL_NAMES)].rsub(held_out[TARGET], axis=0)
    return {
        "Lưới siêu tham số trên train": grid,
        "CV 5 fold trên train": cv_summary,
        "Chỉ số trên test": metrics,
        "Phần dư trên test (g)": residuals.describe().T,
        "Sai số test theo loài và giới tính": _subgroup_errors(held_out),
        "Paired bootstrap trên test": paired_mae_interval(held_out).to_frame("Giá trị"),
        "Hệ số trên thiết kế chuẩn hóa": coefficients,
        "Tổng hợp ảnh hưởng trên train": influence_summary.to_frame("Giá trị"),
        "Quan sát cần đối chiếu trên train": cases,
        "Độ nhạy khi fit lại trên train": sensitivity,
    }, held_out


def plot_test_comparison(predictions):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    actual = predictions[TARGET]
    fitted = predictions[list(MODEL_NAMES)]
    lower = min(actual.min(), fitted.min().min())
    upper = max(actual.max(), fitted.max().max())
    padding = 0.03 * (upper - lower)
    limits = (lower - padding, upper + padding)
    extent = 1.05 * fitted.rsub(actual, axis=0).abs().max().max()
    for name, scatter_ax, residual_ax in zip(MODEL_NAMES, axes[0], axes[1]):
        for (species, group), color in zip(
            predictions.groupby("species"), ("#2874A6", "#C65721", "#16806A"),
        ):
            scatter_ax.scatter(group[TARGET], group[name], s=20, alpha=0.7, color=color, label=species)
            residual_ax.scatter(group[name], group[TARGET] - group[name], s=20, alpha=0.7, color=color)
        scatter_ax.plot(limits, limits, "--", color="#49566c", lw=1)
        scatter_ax.set(xlabel="Thực tế (g)", ylabel="Dự đoán (g)", xlim=limits, ylim=limits, title=name)
        residual_ax.axhline(0, ls="--", color="#49566c", lw=1)
        residual_ax.set(xlabel="Dự đoán (g)", ylabel="Thực tế − dự đoán (g)", xlim=limits, ylim=(-extent, extent))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=len(labels), frameon=False)
    fig.suptitle(f"OLS và M4 Huber · Tập kiểm tra (n = {len(predictions)})")
    return fig


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--figure", type=Path, help="Đường dẫn lưu biểu đồ so sánh trên test.")
    args = parser.parse_args()
    train, test = split_penguins(load_penguins())
    tables, predictions = evaluate_huber(train, test)
    for title, table in tables.items():
        print(f"\n{title}\n{table.to_string(float_format=lambda value: f'{value:.6g}')}")
    if args.figure is not None:
        plt.switch_backend("Agg")
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        figure = plot_test_comparison(predictions)
        figure.savefig(args.figure, dpi=160, facecolor="white")
        plt.close(figure)


if __name__ == "__main__":
    main()
