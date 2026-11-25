"""Train-only Ridge coefficient paths and explanatory design diagnostics."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

from .modeling import CATEGORICAL, FEATURES, TARGET, make_pipeline
from .models.m1 import ALPHA_CANDIDATES

NUMERIC = [feature for feature in FEATURES if feature not in CATEGORICAL]


def train_collinearity(train, ols_model):
    """Pearson on numeric train inputs; auxiliary-regression VIF on full design.

    VIF includes the effects of the one-hot columns. Auxiliary regressions have
    an intercept; the intercept itself is not included in the VIF table.
    This describes X, not Ridge significance and not a different 'Ridge VIF'.
    """
    preprocess = ols_model.named_steps["preprocess"]
    design = np.asarray(preprocess.transform(train[FEATURES]), dtype=float)
    names = list(preprocess.get_feature_names_out())
    pearson = train[NUMERIC].corr(method="pearson").rename_axis("feature").reset_index()
    rows = []
    for index, feature in enumerate(names):
        target = design[:, index]
        if np.ptp(target) == 0:
            raise ValueError(f"VIF is undefined for constant design column {feature}.")
        others = np.delete(design, index, axis=1)
        r_squared = LinearRegression(fit_intercept=True).fit(others, target).score(others, target)
        denominator = 1.0 - r_squared
        vif = np.inf if denominator <= np.finfo(float).eps else 1.0 / denominator
        rows.append({"feature": feature, "R2_auxiliary": r_squared, "VIF": vif,
                     "n_train": len(train), "auxiliary_intercept": True})
    return pearson, pd.DataFrame(rows)


def ridge_coefficient_path(train, ols_model, *, alphas=ALPHA_CANDIDATES):
    """Fit exactly the requested alphas on train; no validation/test selection."""
    alphas = tuple(alphas)
    if not alphas or any(not np.isfinite(alpha) or alpha <= 0 for alpha in alphas):
        raise ValueError("Coefficient path alphas must be positive finite values.")
    reference = ols_model.named_steps["preprocess"]
    names = list(reference.get_feature_names_out())
    reference_design = reference.transform(train[FEATURES])
    ols_coefficients = ols_model.named_steps["regression"].coef_
    path, norms = [], []
    for alpha in alphas:
        pipeline = make_pipeline(Ridge(alpha=alpha)).fit(train[FEATURES], train[TARGET])
        preprocess = pipeline.named_steps["preprocess"]
        if list(preprocess.get_feature_names_out()) != names:
            raise ValueError("Coefficient paths require the same design column order as OLS.")
        np.testing.assert_allclose(preprocess.transform(train[FEATURES]), reference_design, rtol=0, atol=1e-12)
        coefficients = pipeline.named_steps["regression"].coef_
        for feature, coefficient, ols_coefficient in zip(names, coefficients, ols_coefficients, strict=True):
            path.append({"alpha": float(alpha), "feature": feature, "coefficient": coefficient,
                         "OLS": ols_coefficient,
                         "unit": "g per train SD" if feature in NUMERIC else "g vs reference"})
        norms.append({"alpha": float(alpha), "L2_norm": np.linalg.norm(coefficients),
                      "intercept_g": pipeline.named_steps["regression"].intercept_, "n_train": len(train)})
    return pd.DataFrame(path), pd.DataFrame(norms)


def save_coefficient_path(path, selected_alpha, target):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), layout="constrained")
    groups = (("Numeric inputs (g per train SD)", NUMERIC),
              ("Categorical contrasts (g vs reference)", [f for f in path.feature.unique() if f not in NUMERIC]))
    colors = ("#236cb0", "#d56a22", "#238b45")
    for ax, (title, features) in zip(axes, groups, strict=True):
        for color, feature in zip(colors, features, strict=True):
            rows = path.loc[path.feature == feature].sort_values("alpha")
            ax.plot(rows.alpha, rows.coefficient, marker="o", color=color, label=feature)
            ax.axhline(rows.OLS.iloc[0], color=color, linestyle="--", alpha=0.55, linewidth=1)
        ax.axvline(selected_alpha, color="#444444", linestyle=":", label=f"Selected alpha={selected_alpha:g}")
        ax.set_xscale("log")
        alphas = sorted(path.alpha.unique())
        ax.set_xticks(alphas, [f"{alpha:g}" for alpha in alphas])
        ax.set(title=title, xlabel="Ridge alpha (log scale)", ylabel="Coefficient")
        ax.grid(alpha=0.15)
        ax.legend(fontsize=8)
    fig.suptitle("Train-only coefficient paths: solid=Ridge, dashed=OLS; intercept excluded")
    fig.savefig(target, dpi=150)
    plt.close(fig)
