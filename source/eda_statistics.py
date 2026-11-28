"""Statistical EDA on the shared Palmer Penguins training split."""

from argparse import ArgumentParser
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold, cross_validate
from statsmodels.formula.api import ols
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway

from .modeling import FEATURES, TARGET, load_penguins, make_pipeline, split_penguins

MEASUREMENTS = ("flipper_length_mm", "bill_length_mm", "bill_depth_mm")
DEPTH = "bill_depth_mm"
SEED = 42


def correlation_tables(train):
    rng = np.random.default_rng(SEED)
    permutation = stats.PermutationMethod(n_resamples=9_999, batch=128, rng=rng)
    bootstrap = stats.BootstrapMethod(n_resamples=2_000, method="percentile", batch=128, rng=rng)
    pearson_rows = []
    spearman_rows = []
    for label, group in [("Gộp", train), *train.groupby("species")]:
        for feature in MEASUREMENTS:
            pairs = group[[feature, TARGET]].dropna()
            x, y = pairs[feature], pairs[TARGET]
            result = stats.pearsonr(x, y, method=permutation)
            ci = result.confidence_interval(0.95, method=bootstrap)
            context = {"Nhóm": label, "Số đo": feature, "n": len(pairs)}
            pearson_rows.append({
                **context, "Pearson r": result.statistic, "p hoán vị": result.pvalue,
                "CI 95% dưới": ci.low, "CI 95% trên": ci.high,
            })
            if label == "Gộp" or feature == DEPTH:
                spearman_rows.append({
                    **context, "Spearman ρ": stats.spearmanr(x, y).statistic,
                })
    return pd.DataFrame(pearson_rows), pd.DataFrame(spearman_rows)


def _joint_test(fit, term):
    row = fit.wald_test_terms(scalar=True).table.loc[term]
    return {
        "n": int(fit.nobs), "F": float(row["statistic"]),
        "df1": int(row["df_constraint"]), "df2": float(row["df_denom"]),
        "p": float(row["pvalue"]),
    }


def depth_analysis(train):
    formula = f"{TARGET} ~ {DEPTH}"
    rows = []
    fits = [
        ("Gộp", formula, train),
        ("Điều chỉnh theo loài", f"{formula} + C(species)", train),
        *((label, formula, group) for label, group in train.groupby("species")),
    ]
    for label, specification, data in fits:
        fit = ols(specification, data=data).fit(cov_type="HC3", use_t=True)
        lower, upper = fit.conf_int().loc[DEPTH]
        rows.append({
            "Mô hình phụ": label, "n": int(fit.nobs), "Hệ số (g/mm)": fit.params[DEPTH],
            "t HC3": fit.tvalues[DEPTH], "p HC3": fit.pvalues[DEPTH],
            "CI 95% dưới": lower, "CI 95% trên": upper,
        })

    interacted = ols(f"{formula} * C(species)", data=train).fit(cov_type="HC3", use_t=True)
    test = _joint_test(interacted, f"{DEPTH}:C(species)")
    return pd.DataFrame(rows), {"Kiểm định": "Độ sâu mỏ × loài", **test}


def _welch_differences(comparisons):
    rows = []
    for label, first, second in comparisons:
        result = stats.ttest_ind(first, second, equal_var=False)
        ci = result.confidence_interval(0.95)
        rows.append({
            "So sánh": label, "n1": len(first), "n2": len(second),
            "Chênh lệch (g)": first.mean() - second.mean(),
            "CI 95% dưới": ci.low, "CI 95% trên": ci.high,
            "t Welch": result.statistic, "df": result.df, "p": result.pvalue,
        })
    table = pd.DataFrame(rows)
    table["p Holm"] = multipletests(table["p"], method="holm")[1]
    return table


def _mass_summary(grouped):
    summary = grouped.agg(
        n="size", mean_g="mean", sd_g="std", median_g="median",
        q1_g=lambda x: x.quantile(0.25), q3_g=lambda x: x.quantile(0.75),
    )
    summary["iqr_g"] = summary["q3_g"] - summary["q1_g"]
    return summary


def group_analysis(train):
    observed_sex = train.dropna(subset=["sex"])
    mass_by_species = train.groupby("species")[TARGET]
    species_groups = list(mass_by_species)
    anova = anova_oneway([mass for _, mass in species_groups], use_var="unequal")
    overall = {
        "Kiểm định": "Welch ANOVA theo loài", "n": int(mass_by_species.count().sum()),
        "F": anova.statistic, "df1": int(anova.df_num), "df2": anova.df_denom, "p": anova.pvalue,
    }
    descriptives = pd.concat([
        _mass_summary(mass_by_species).assign(sex="Tất cả").set_index("sex", append=True),
        _mass_summary(observed_sex.groupby(["species", "sex"])[TARGET]),
    ]).sort_index()

    species_contrasts = _welch_differences([
        (f"{name2} − {name1}", mass2, mass1)
        for (name1, mass1), (name2, mass2) in combinations(species_groups, 2)
    ])
    sex_contrasts = _welch_differences([
        (f"{name}: đực − cái", group.loc[group["sex"].eq("male"), TARGET],
         group.loc[group["sex"].eq("female"), TARGET])
        for name, group in observed_sex.groupby("species")
    ])

    sex_fit = ols(f"{TARGET} ~ C(species) * C(sex)", data=observed_sex).fit(
        cov_type="HC3", use_t=True,
    )
    sex_test = {"Kiểm định": "Loài × giới tính", **_joint_test(sex_fit, "C(species):C(sex)")}
    interaction_effects = sex_fit.conf_int().assign(
        difference_g=sex_fit.params, t_hc3=sex_fit.tvalues, p_hc3=sex_fit.pvalues,
    ).rename(columns={0: "ci_low_g", 1: "ci_high_g"})
    interaction_effects = interaction_effects.loc[
        interaction_effects.index.str.contains(":", regex=False),
        ["difference_g", "ci_low_g", "ci_high_g", "t_hc3", "p_hc3"],
    ].copy()
    interaction_effects["p_holm"] = multipletests(interaction_effects["p_hc3"], method="holm")[1]
    return descriptives, species_contrasts, sex_contrasts, overall, sex_test, interaction_effects


def distribution_audit(train):
    summaries = []
    flags = []
    for label, group in [("Gộp", train), *train.groupby("species")]:
        for feature in (*MEASUREMENTS, TARGET):
            values = group[feature].dropna()
            q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            flagged = values[~values.between(lower, upper)]
            summaries.append({
                "Nhóm": label, "Biến": feature, "n": len(values), "SD": values.std(),
                "Trung vị": median, "IQR": iqr, "Skewness": values.skew(),
                "Cận dưới": lower, "Cận trên": upper, "Số điểm gắn cờ": len(flagged),
            })
            flags.extend(
                {"Nhóm": label, "Biến": feature, "Chỉ số CSV": index,
                 "Loài": group.at[index, "species"], "Giá trị": value}
                for index, value in flagged.items()
            )
    return pd.DataFrame(summaries), pd.DataFrame(
        flags, columns=["Nhóm", "Biến", "Chỉ số CSV", "Loài", "Giá trị"],
    )


def feature_cv(train):
    """Refit the shared OLS pipeline on identical folds for each feature set."""
    folds = list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(train))
    scoring = {
        "MAE": "neg_mean_absolute_error", "MSE": "neg_mean_squared_error",
        "RMSE": "neg_root_mean_squared_error", "R²": "r2",
    }
    configurations = [
        ("Đủ 5 biến", FEATURES),
        ("Chỉ ba số đo", list(MEASUREMENTS)),
        *((f"Bỏ {feature}", [column for column in FEATURES if column != feature])
          for feature in FEATURES),
    ]
    rows = []
    for label, columns in configurations:
        results = cross_validate(
            make_pipeline(features=columns), train[columns], train[TARGET],
            cv=folds, scoring=scoring, error_score="raise",
        )
        row = {"Bộ biến": label}
        for metric in scoring:
            scores = results[f"test_{metric}"]
            if metric != "R²":
                scores = -scores
            row[f"{metric} TB"] = scores.mean()
            row[f"{metric} SD"] = scores.std(ddof=1)
        rows.append(row)
    table = pd.DataFrame(rows).set_index("Bộ biến")
    table["Δ MAE so với đủ 5 biến (g)"] = table["MAE TB"] - table.loc["Đủ 5 biến", "MAE TB"]
    return table


def analyze_eda(train):
    pearson, spearman = correlation_tables(train)
    slopes, depth_interaction = depth_analysis(train)
    summary, species, sex, anova, sex_interaction, interaction_effects = group_analysis(train)
    distribution, flags = distribution_audit(train)
    return {
        "Pearson và CI bootstrap": pearson,
        "Đối chiếu Spearman": spearman,
        "Hệ số độ sâu mỏ (HC3)": slopes,
        "Kiểm định tổng thể": pd.DataFrame([depth_interaction, anova, sex_interaction]),
        "Khối lượng theo loài và giới tính": summary,
        "Chênh lệch giữa các loài": species,
        "Chênh lệch giới tính trong từng loài": sex,
        "Thay đổi chênh lệch giới tính so với Adelie (HC3)": interaction_effects,
        "Phân bố và ngưỡng IQR": distribution,
        "Quan sát gắn cờ IQR": flags,
        "Tương quan giữa ba số đo": train[list(MEASUREMENTS)].corr(),
        "So sánh bộ biến qua CV": feature_cv(train),
    }


def plot_lowess(train):
    labels = ("Chiều dài cánh bơi", "Chiều dài mỏ", "Độ sâu mỏ")
    colors = ("#2874A6", "#C65721", "#16806A")
    species_groups = list(train.groupby("species"))
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True, layout="constrained")
    for column, (feature, label) in enumerate(zip(MEASUREMENTS, labels)):
        for row, ax in enumerate(axes[:, column]):
            for (name, group), color in zip(species_groups, colors):
                ax.scatter(group[feature], group[TARGET], color=color, s=12, alpha=0.22)
                if row:
                    curve = lowess(group[TARGET], group[feature], frac=0.65, it=3)
                    ax.plot(*curve.T, color=color, label=name, linewidth=2)
            if row == 0:
                curve = lowess(train[TARGET], train[feature], frac=0.65, it=3)
                ax.plot(*curve.T, color="#263549", linewidth=2.2)
            ax.set_title(label)
            ax.set_xlabel("mm")
            ax.grid(axis="y", alpha=0.2)
    axes[0, 0].set_ylabel("Khối lượng (g) · Gộp")
    axes[1, 0].set_ylabel("Khối lượng (g) · Theo loài")
    handles, legend = axes[1, 0].get_legend_handles_labels()
    figure.legend(handles, legend, loc="outside lower center", ncols=3, frameon=False)
    figure.suptitle(f"LOWESS gộp và theo loài · Tập train (n = {len(train)})")
    return figure


def main():
    parser = ArgumentParser(description="Chạy EDA bổ sung trên tập train.")
    parser.add_argument("--figure", type=Path, help="Đường dẫn xuất biểu đồ LOWESS.")
    args = parser.parse_args()
    train, _ = split_penguins(load_penguins())
    for name, table in analyze_eda(train).items():
        print(f"\n{name}\n{table.to_string(float_format=lambda value: f'{value:.5g}')}")
    if args.figure is not None:
        plt.switch_backend("Agg")
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        figure = plot_lowess(train)
        figure.savefig(args.figure, dpi=170, facecolor="white")
        plt.close(figure)


if __name__ == "__main__":
    main()
