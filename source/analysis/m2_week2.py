"""M2 – Cubic Spline Basis Regression · phân tích tuần 2 (Issue #11).

Chạy từ thư mục gốc dự án:
    python -m source.analysis.m2_week2

Script KHÔNG sửa code dùng chung hay code của thành viên khác: chỉ import
`source.modeling`, `source.models.m2` và `source.models.m3` để vẽ thêm đường so sánh.
Mọi kết quả ghi vào `report/m2_week2/`. Test set chỉ dùng để báo cáo, không dùng để chọn siêu tham số.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, cross_validate

from source.modeling import CATEGORICAL, FEATURES, TARGET, load_penguins, make_pipeline, split_penguins
from source.models import m2, m3

warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42
N_BOOT = 2000          # paired bootstrap (Issue #11)
N_CURVE_BOOT = 500     # bootstrap refit để vẽ dải bất định của đường cong
N_NULL_BOOT = 1000     # residual bootstrap dưới H0 tuyến tính
KNOT_GRID = m2.KNOT_GRID
NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]

OUT = Path(__file__).resolve().parents[2] / "report" / "m2_week2"
FIG, TAB = OUT / "figures", OUT / "tables"
for folder in (FIG, TAB):
    folder.mkdir(parents=True, exist_ok=True)

LABELS = {
    "flipper_length_mm": "Chiều dài cánh bơi (mm)",
    "bill_length_mm": "Chiều dài mỏ (mm)",
    "bill_depth_mm": "Độ sâu mỏ (mm)",
}
SPECIES_COLORS = {"Adelie": "#2874A6", "Chinstrap": "#C65721", "Gentoo": "#16806A"}
C_M2, C_OLS, C_M3 = "#C65721", "#2874A6", "#16806A"
RESULTS: dict = {}


def jsonable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.to_dict()
    raise TypeError(type(obj))


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def metrics(y, p):
    mse = mean_squared_error(y, p)
    return {"MAE (g)": mean_absolute_error(y, p), "MSE (g²)": mse, "RMSE (g)": float(np.sqrt(mse)), "R²": r2_score(y, p)}


def build_ols():
    return make_pipeline(LinearRegression())  # cùng preprocessing (impute + StandardScaler + one-hot) như pipeline chung


def build_m2(n_knots=None):
    model = m2.build_model()
    if n_knots is not None:
        model.set_params(preprocess__numeric__scaler__n_knots=n_knots)
    return model


SCORING = {"MAE": "neg_mean_absolute_error", "MSE": "neg_mean_squared_error", "R2": "r2"}


def cv_scores(model, X, y, cv):
    r = cross_validate(model, X, y, cv=cv, scoring=SCORING, return_estimator=True)
    mse = -r["test_MSE"]
    frame = pd.DataFrame({
        "fold": np.arange(1, len(mse) + 1), "MAE (g)": -r["test_MAE"], "MSE (g²)": mse,
        "RMSE (g)": np.sqrt(mse), "R²": r["test_R2"],
    })
    return frame, r["estimator"]


def summarize_cv(frame):
    cols = ["MAE (g)", "MSE (g²)", "RMSE (g)", "R²"]
    return pd.DataFrame({"mean": frame[cols].mean(), "std (ddof=1)": frame[cols].std(ddof=1)}).T


def savefig(fig, name):
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
def main():
    penguins = load_penguins()
    train, test = split_penguins(penguins)
    Xtr, ytr, Xte, yte = train[FEATURES], train[TARGET], test[FEATURES], test[TARGET]
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    RESULTS["setup"] = {"n_train": len(train), "n_test": len(test), "seed": SEED, "n_boot": N_BOOT}

    # ============================ A. Kiểm tra tính đúng đắn ============================
    section("A. KIỂM TRA CẤU HÌNH, SỐ ĐẶC TRƯNG VÀ RÒ RỈ DỮ LIỆU")
    m2_full = build_m2().fit(Xtr, ytr)
    spline = m2_full.named_steps["preprocess"].named_transformers_["numeric"].named_steps["scaler"]
    cfg = {k: spline.get_params()[k] for k in ("degree", "n_knots", "knots", "include_bias", "extrapolation")}
    print("Cấu hình SplineTransformer:", cfg)
    assert cfg == {"degree": 3, "n_knots": 3, "knots": "uniform", "include_bias": False, "extrapolation": "constant"}
    print("Estimator cuối:", type(m2_full.named_steps["regression"]).__name__)
    Z = m2_full.named_steps["preprocess"].transform(Xtr)
    names = list(m2_full.named_steps["preprocess"].get_feature_names_out())
    n_spline = sum("_sp_" in n for n in names)
    print(f"Hình dạng đặc trưng sau biến đổi: {Z.shape} | spline: {n_spline} | dummy: {Z.shape[1] - n_spline}")
    assert Z.shape == (273, 15) and n_spline == 12

    # Reproduce số liệu tuần 1
    pred_te_check = m2_full.predict(Xte)
    base = metrics(yte, pred_te_check)
    assert abs(base["MAE (g)"] - 252.934) < 5e-4 and abs(base["RMSE (g)"] - 308.237) < 5e-4 and abs(base["R²"] - 0.832) < 5e-4
    print("Tái hiện test tuần 1 (MAE 252.934, RMSE 308.237, R² 0.832): ĐẠT")

    # Fit-trong-fold: kiểm tra từng fold
    fold_frame, fold_models = cv_scores(build_m2(), Xtr, ytr, cv)
    leak_rows = []
    for k, ((tr_idx, va_idx), est) in enumerate(zip(cv.split(Xtr), fold_models), start=1):
        sub = Xtr.iloc[tr_idx]
        pre = est.named_steps["preprocess"]
        imp = pre.named_transformers_["numeric"].named_steps["imputer"]
        sp = pre.named_transformers_["numeric"].named_steps["scaler"]
        np.testing.assert_allclose(imp.statistics_, sub[NUMERIC].median().to_numpy())
        for j, f in enumerate(NUMERIC):
            t = sp.bsplines_[j].t
            assert np.isclose(t[3], sub[f].min()) and np.isclose(t[-4], sub[f].max())
        cat_imp = pre.named_transformers_["categorical"].named_steps["imputer"]
        assert list(cat_imp.statistics_) == [sub[c].mode().iloc[0] for c in CATEGORICAL if c in FEATURES]
        va = Xtr.iloc[va_idx]
        outside = int(sum(((va[f] < sub[f].min()) | (va[f] > sub[f].max())).sum() for f in NUMERIC))
        leak_rows.append({"fold": k, "n_fit": len(tr_idx), "n_valid": len(va_idx),
                          "median_imputer_dùng_fold_train": True, "knots_dùng_min/max_fold_train": True,
                          "giá_trị_valid_ngoài_miền_fold_train": outside})
    leak = pd.DataFrame(leak_rows)
    print(leak.to_string(index=False))
    full_knots = np.array([sp_t[3] for sp_t in [b.t for b in spline.bsplines_]])
    print("Mốc biên spline (min) khi fit toàn bộ train:", dict(zip(NUMERIC, full_knots)))
    print("=> Imputer + SplineTransformer + OneHot được fit lại trong mỗi fold (cross_validate clone cả Pipeline).")
    leak.to_csv(TAB / "A_leakage_check_per_fold.csv", index=False)
    out_of_range_test = {f: int(((Xte[f] < Xtr[f].min()) | (Xte[f] > Xtr[f].max())).sum()) for f in NUMERIC}
    print("Số quan sát test nằm ngoài miền train (cần extrapolation):", out_of_range_test)
    RESULTS["A"] = {"config": cfg, "transformed_shape": list(Z.shape), "feature_names": names,
                    "test_out_of_train_range": out_of_range_test, "week1_reproduced": True,
                    "missing_in_train": train[FEATURES].isna().sum().to_dict()}

    # ============================ B. Hiệu suất ============================
    section("B. METRICS TEST, 5-FOLD CV, CHỌN n_knots (CHỈ TRÊN TRAIN)")
    ols = build_ols().fit(Xtr, ytr)
    p2, po = m2_full.predict(Xte), ols.predict(Xte)
    test_tbl = pd.DataFrame({"M2 (Spline)": metrics(yte, p2), "OLS": metrics(yte, po)}).T
    print(test_tbl.round(3).to_string())
    test_tbl.to_csv(TAB / "B_test_metrics.csv")

    cv_m2, _ = cv_scores(build_m2(), Xtr, ytr, cv)
    cv_ols, _ = cv_scores(build_ols(), Xtr, ytr, cv)
    cv_m2.assign(model="M2").pipe(lambda d: pd.concat([d, cv_ols.assign(model="OLS")])).to_csv(TAB / "B_cv_per_fold.csv", index=False)
    cv_tbl = pd.concat({"M2 (Spline)": summarize_cv(cv_m2), "OLS": summarize_cv(cv_ols)})
    print("\n5-fold CV trên train (RMSE = trung bình của RMSE từng fold):")
    print(cv_tbl.round(3).to_string())
    cv_tbl.to_csv(TAB / "B_cv_summary.csv")
    diff = cv_m2[["MAE (g)", "MSE (g²)", "RMSE (g)", "R²"]] - cv_ols[["MAE (g)", "MSE (g²)", "RMSE (g)", "R²"]]
    print("\nChênh lệch theo fold (M2 − OLS):")
    print(diff.round(3).assign(fold=range(1, 6)).to_string(index=False))
    print("MAE: M2 tốt hơn OLS ở", int((diff["MAE (g)"] < 0).sum()), "/ 5 fold")

    knots_rows = []
    for k in KNOT_GRID:
        f, _ = cv_scores(build_m2(k), Xtr, ytr, cv)
        knots_rows.append({"n_knots": k, **summarize_cv(f).loc["mean"].to_dict(), "MAE std": f["MAE (g)"].std(ddof=1),
                           "n_features": 3 * (k + 3 - 1 - 1) + 3})
    knots_tbl = pd.DataFrame(knots_rows)
    print("\nChọn n_knots bằng 5-fold CV trên train (tái hiện bảng tuần 1):")
    print(knots_tbl.round(3).to_string(index=False))
    knots_tbl.to_csv(TAB / "B_nknots_cv.csv", index=False)
    assert abs(knots_tbl.loc[knots_tbl.n_knots == 3, "MAE (g)"].iloc[0] - 234.404) < 5e-4

    # Nested CV: chọn n_knots trong vòng trong -> ước lượng không lạc quan do chọn siêu tham số
    grid = GridSearchCV(build_m2(), {"preprocess__numeric__scaler__n_knots": list(KNOT_GRID)},
                        cv=KFold(5, shuffle=True, random_state=SEED), scoring="neg_mean_absolute_error")
    nested = cross_validate(grid, Xtr, ytr, cv=cv, scoring=SCORING, return_estimator=True)
    chosen = [e.best_params_["preprocess__numeric__scaler__n_knots"] for e in nested["estimator"]]
    nested_tbl = pd.DataFrame({"fold": range(1, 6), "n_knots được chọn": chosen, "MAE (g)": -nested["test_MAE"],
                               "RMSE (g)": np.sqrt(-nested["test_MSE"]), "R²": nested["test_R2"]})
    print("\nNested CV (outer 5-fold, inner 5-fold chọn n_knots):")
    print(nested_tbl.round(3).to_string(index=False))
    print(f"MAE nested = {nested_tbl['MAE (g)'].mean():.3f} ± {nested_tbl['MAE (g)'].std(ddof=1):.3f} | "
          f"OLS CV = {cv_ols['MAE (g)'].mean():.3f} ± {cv_ols['MAE (g)'].std(ddof=1):.3f}")
    nested_tbl.to_csv(TAB / "B_nested_cv.csv", index=False)
    RESULTS["B"] = {"test": test_tbl.to_dict("index"), "cv": {"M2": summarize_cv(cv_m2).to_dict("index"), "OLS": summarize_cv(cv_ols).to_dict("index")},
                    "nested_mae_mean": nested_tbl["MAE (g)"].mean(), "nested_mae_std": nested_tbl["MAE (g)"].std(ddof=1),
                    "nested_chosen": chosen, "folds_m2_better_mae": int((diff["MAE (g)"] < 0).sum())}

    # ---- sai số theo nhóm
    def group_table(df, by):
        d = df.copy()
        d["sex"] = d["sex"].fillna("thiếu")
        d["res"] = d["y"] - d["pred"]
        out = d.groupby(by).agg(n=("res", "size"), MAE=("res", lambda r: r.abs().mean()),
                                RMSE=("res", lambda r: np.sqrt((r ** 2).mean())), bias=("res", "mean")).round(2)
        return out

    test_frame = pd.DataFrame({"species": test["species"], "sex": test["sex"], "y": yte, "pred": p2})
    oof = cross_val_predict(build_m2(), Xtr, ytr, cv=cv)
    oof_frame = pd.DataFrame({"species": train["species"], "sex": train["sex"], "y": ytr, "pred": oof})
    for tag, frame in (("test", test_frame), ("oof_train", oof_frame)):
        parts = [group_table(frame, "species").assign(nhóm="species"), group_table(frame, "sex").assign(nhóm="sex"),
                 group_table(frame, ["species", "sex"]).assign(nhóm="species×sex")]
        tbl = pd.concat([p.reset_index().rename(columns={"index": "giá trị"}) for p in parts])
        tbl.to_csv(TAB / f"B_group_errors_{tag}.csv", index=False)
        print(f"\nSai số M2 theo nhóm – {tag} (bias = trung bình thực tế − dự đoán):")
        for p_ in parts:
            print("\n".join(line.rstrip() for line in p_.to_string().splitlines()))
    RESULTS["B"]["oof_overall"] = metrics(ytr, oof)

    # ---- biểu đồ actual vs predicted, residual
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained", sharex=True, sharey=True)
    lims = [min(yte.min(), p2.min(), po.min()) - 100, max(yte.max(), p2.max(), po.max()) + 100]
    for ax, pred, name in zip(axes, (p2, po), ("M2 – Cubic Spline", "OLS")):
        for sp_name, c in SPECIES_COLORS.items():
            m = (test["species"] == sp_name).to_numpy()
            ax.scatter(pred[m], yte[m], s=26, alpha=0.75, color=c, label=f"{sp_name} (n={m.sum()})")
        ax.plot(lims, lims, "--", color="#475569", lw=1)
        s = metrics(yte, pred)
        ax.set(title=f"{name}  |  MAE {s['MAE (g)']:.1f} g · RMSE {s['RMSE (g)']:.1f} g · R² {s['R²']:.3f}",
               xlabel="Dự đoán (g)", xlim=lims, ylim=lims)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Thực tế (g)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Thực tế so với dự đoán · tập kiểm tra (n = 69)")
    savefig(fig, "fig01_actual_vs_predicted.png")

    res_te, res_oof, res_ols_te = yte - p2, ytr - oof, yte - po
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for ax, pred, res, frame, ttl in (
        (axes[0, 0], p2, res_te, test, "M2 · phần dư trên test (n = 69)"),
        (axes[0, 1], po, res_ols_te, test, "OLS · phần dư trên test (n = 69)"),
        (axes[1, 0], oof, res_oof, train, "M2 · phần dư out-of-fold trên train (n = 273)"),
    ):
        for sp_name, c in SPECIES_COLORS.items():
            m = (frame["species"] == sp_name).to_numpy()
            ax.scatter(np.asarray(pred)[m], np.asarray(res)[m], s=22, alpha=0.7, color=c, label=sp_name)
        ax.axhline(0, ls="--", color="#475569")
        ax.set(title=ttl, xlabel="Dự đoán (g)", ylabel="Thực tế − dự đoán (g)")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=8)
    stats.probplot(res_oof, dist="norm", plot=axes[1, 1])
    axes[1, 1].set(title="Q–Q plot phần dư out-of-fold của M2", xlabel="Phân vị lý thuyết", ylabel="Phân vị mẫu (g)")
    axes[1, 1].get_lines()[0].set(markerfacecolor=C_M2, markeredgecolor=C_M2, markersize=3, alpha=0.7)
    savefig(fig, "fig02_residuals.png")
    RESULTS["B"]["shapiro_oof_resid"] = dict(zip(("W", "p"), map(float, stats.shapiro(res_oof))))
    RESULTS["B"]["levene_oof_resid_by_species"] = dict(zip(("stat", "p"), map(float, stats.levene(
        *[res_oof[(train["species"] == s).to_numpy()] for s in SPECIES_COLORS]))))
    print("\nShapiro (phần dư OOF M2):", RESULTS["B"]["shapiro_oof_resid"])
    print("Levene theo loài (phần dư OOF M2):", RESULTS["B"]["levene_oof_resid_by_species"])

    # ============================ D. Paired bootstrap ============================
    section("D. PAIRED BOOTSTRAP M2 vs OLS (69 mẫu test, 2000 lần, seed 42)")
    rng = np.random.default_rng(SEED)
    n = len(yte)
    idx = rng.integers(0, n, size=(N_BOOT, n))   # CÙNG chỉ số cho M2 và OLS -> paired, không refit
    y_arr, a2, ao = yte.to_numpy(), np.abs(yte.to_numpy() - p2), np.abs(yte.to_numpy() - po)
    d_mae = a2[idx].mean(1) - ao[idx].mean(1)
    s2, so = (y_arr - p2) ** 2, (y_arr - po) ** 2
    d_rmse = np.sqrt(s2[idx].mean(1)) - np.sqrt(so[idx].mean(1))
    sst = ((y_arr[idx] - y_arr[idx].mean(1, keepdims=True)) ** 2).sum(1)
    d_r2 = (1 - s2[idx].sum(1) / sst) - (1 - so[idx].sum(1) / sst)
    boot_rows = []
    for name, d, obs in (
        ("ΔMAE (g)  [chỉ số chính]", d_mae, a2.mean() - ao.mean()),
        ("ΔRMSE (g) [phụ, khám phá]", d_rmse, np.sqrt(s2.mean()) - np.sqrt(so.mean())),
        ("ΔR²       [phụ, khám phá]", d_r2, r2_score(y_arr, p2) - r2_score(y_arr, po)),
    ):
        lo, hi = np.percentile(d, [2.5, 97.5])
        boot_rows.append({"chỉ số": name, "quan sát": obs, "trung bình bootstrap": d.mean(), "CI95 thấp": lo, "CI95 cao": hi,
                          "CI chứa 0": bool(lo <= 0 <= hi), "tỷ lệ Δ<0": float((d < 0).mean())})
    boot_tbl = pd.DataFrame(boot_rows)
    print(boot_tbl.round(3).to_string(index=False))
    boot_tbl.to_csv(TAB / "D_paired_bootstrap.csv", index=False)
    print(f"MAE test: M2 {a2.mean():.3f} | OLS {ao.mean():.3f} | số mẫu M2 sai số tuyệt đối nhỏ hơn OLS: {(a2 < ao).sum()}/{n}")
    RESULTS["D"] = boot_tbl.to_dict("records")
    RESULTS["D"].append({"mae_m2": a2.mean(), "mae_ols": ao.mean(), "n_m2_better": int((a2 < ao).sum()), "n": n})

    fig, ax = plt.subplots(figsize=(7.5, 4.4), layout="constrained")
    ax.hist(d_mae, bins=50, color="#8FB8D6", edgecolor="white")
    lo, hi = np.percentile(d_mae, [2.5, 97.5])
    ax.axvline(0, color="k", ls="--", label="Δ = 0 (không khác biệt)")
    ax.axvline(a2.mean() - ao.mean(), color=C_M2, lw=2, label=f"Quan sát = {a2.mean() - ao.mean():+.2f} g")
    ax.axvspan(lo, hi, color=C_M2, alpha=0.12, label=f"CI 95% = [{lo:+.2f}; {hi:+.2f}] g")
    ax.set(xlabel="ΔMAE = MAE(M2) − MAE(OLS)  (g)", ylabel="Số lần bootstrap",
           title="Paired bootstrap · 2.000 lần · 69 mẫu test")
    ax.legend(frameon=False, fontsize=9)
    savefig(fig, "fig03_paired_bootstrap_delta_mae.png")

    # ============================ C. Conditional prediction ============================
    section("C. CONDITIONAL PREDICTION (giữ các biến khác cố định, không phải quan hệ nhân quả)")
    ref = {f: float(train[f].median()) for f in NUMERIC}
    ref.update({c: train[c].mode().iloc[0] for c in CATEGORICAL if c in FEATURES})
    print("Giá trị tham chiếu (median / mode của TRAIN):", ref)
    RESULTS["C"] = {"reference": ref}

    models = {"M2": m2_full, "OLS": ols, "M3": m3.build_model().fit(Xtr, ytr)}
    print("Đã nạp M3 và fit trên cùng 273 mẫu train.")
    RESULTS["C"]["m3_loaded"] = True

    def make_frame(base, var, values):
        df = pd.DataFrame([base] * len(values))
        df[var] = values
        return df[FEATURES]

    grids, lims_ = {}, {}
    for f in NUMERIC:
        lo_, hi_ = float(train[f].min()), float(train[f].max())
        pad = 0.10 * (hi_ - lo_)
        grids[f] = np.linspace(lo_ - pad, hi_ + pad, 241)
        lims_[f] = (lo_, hi_, float(train[f].quantile(0.05)), float(train[f].quantile(0.95)))
    curves = {name: {f: m.predict(make_frame(ref, f, grids[f])) for f in NUMERIC} for name, m in models.items()}

    # Bootstrap refit -> dải bất định pointwise cho M2 và OLS
    rng_c = np.random.default_rng(SEED)
    store = {(nm, f): [] for nm in ("M2", "OLS") for f in NUMERIC}
    for _ in range(N_CURVE_BOOT):
        bt = train.iloc[rng_c.integers(0, len(train), len(train))]
        for nm, builder in (("M2", build_m2), ("OLS", build_ols)):
            mb = builder().fit(bt[FEATURES], bt[TARGET])
            for f in NUMERIC:
                store[(nm, f)].append(mb.predict(make_frame(ref, f, grids[f])))
    bands = {k: np.percentile(np.array(v), [2.5, 97.5], axis=0) for k, v in store.items()}

    cover_rows = []
    for f in NUMERIC:
        _, _, p5_, p95_ = lims_[f]
        inside = (grids[f] >= p5_) & (grids[f] <= p95_)
        lo_b, hi_b = bands[("M2", f)]
        ols_in = (curves["OLS"][f] >= lo_b) & (curves["OLS"][f] <= hi_b)
        cover_rows.append({"biến": f, "tỷ lệ điểm p5–p95 mà đường OLS nằm trong dải CI95 của M2": float(ols_in[inside].mean()),
                           "|M2 − OLS| lớn nhất trong p5–p95 (g)": float(np.abs(curves["M2"][f] - curves["OLS"][f])[inside].max()),
                           "nửa bề rộng dải CI95 M2, trung bình (g)": float(((hi_b - lo_b) / 2)[inside].mean())})
    cover_tbl = pd.DataFrame(cover_rows)
    print(cover_tbl.round(3).to_string(index=False))
    cover_tbl.to_csv(TAB / "C_band_coverage.csv", index=False)
    RESULTS["C"]["band_coverage"] = cover_tbl.to_dict("records")

    def amplitude(x, fx, lo, hi):
        m = (x >= lo) & (x <= hi)
        coef = np.polyfit(x[m], fx[m], 1)
        dev = fx[m] - np.polyval(coef, x[m])
        return float(np.abs(dev).max()), float(x[m][np.abs(dev).argmax()]), coef[0]

    curv_rows = []
    for f in NUMERIC:
        lo_, hi_, p5, p95 = lims_[f]
        for nm in models:
            amp, at, slope = amplitude(grids[f], curves[nm][f], p5, p95)
            third = np.linspace(p5, p95, 4)
            slopes = []
            for a, b in zip(third[:-1], third[1:]):
                mm = (grids[f] >= a) & (grids[f] <= b)
                slopes.append(float(np.polyfit(grids[f][mm], curves[nm][f][mm], 1)[0]))
            curv_rows.append({"biến": f, "mô hình": nm, "biên độ lệch khỏi đường thẳng (g)": amp, "tại x (mm)": at,
                              "độ dốc TB (g/mm)": slope, "dốc đoạn thấp": slopes[0], "dốc đoạn giữa": slopes[1],
                              "dốc đoạn cao": slopes[2], "khoảng p5–p95 (mm)": f"{p5:.1f}–{p95:.1f}",
                              "tổng thay đổi theo đường thẳng p5→p95 (g)": slope * (p95 - p5)})
    curv_tbl = pd.DataFrame(curv_rows)
    print(curv_tbl.round(2).to_string(index=False))
    curv_tbl.to_csv(TAB / "C_curvature_summary.csv", index=False)
    RESULTS["C"]["curvature"] = curv_tbl.to_dict("records")

    for f in NUMERIC:
        lo_, hi_, p5, p95 = lims_[f]
        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        ax.axvspan(grids[f][0], lo_, color="0.92"); ax.axvspan(hi_, grids[f][-1], color="0.92")
        for nm, c, ls in (("OLS", C_OLS, "-"), ("M2", C_M2, "-"), ("M3", C_M3, "--")):
            if nm not in curves:
                continue
            lab = {"OLS": "OLS", "M2": "M2 – Cubic Spline", "M3": "M3 – Smoothing spline"}[nm]
            ax.plot(grids[f], curves[nm][f], color=c, ls=ls, lw=2, label=lab)
            if (nm, f) in bands:
                ax.fill_between(grids[f], *bands[(nm, f)], color=c, alpha=0.13)
        ax.plot(train[f], np.full(len(train), 0.015), "|", color="k", alpha=0.35, transform=ax.get_xaxis_transform())
        ax.axvline(p5, color="0.5", lw=0.8, ls=":"); ax.axvline(p95, color="0.5", lw=0.8, ls=":")
        ax.set(xlabel=LABELS[f], ylabel="Khối lượng dự đoán (g)",
               title=f"Dự đoán có điều kiện theo {LABELS[f].lower()}")
        ref_txt = ", ".join(f"{LABELS[g].split(' (')[0].lower()} = {ref[g]:.1f}" for g in NUMERIC if g != f)
        ax.text(0.01, 0.99, f"Giữ cố định: {ref_txt}, loài = {ref['species']}, giới tính = {ref['sex']}",
                transform=ax.transAxes, va="top", fontsize=8, color="0.3")
        ax.text(0.99, 0.065, "vùng xám: ngoài miền train (extrapolation) · dải: CI 95% bootstrap (500 lần refit)",
                transform=ax.transAxes, ha="right", fontsize=7.5, color="0.35")
        ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 0.11))
        ax.grid(alpha=0.25)
        savefig(fig, f"fig04_conditional_{f}.png")

    # Phiên bản "bám miền dữ liệu": từng loài, biến khác ở median của loài đó
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), layout="constrained")
    for r_, sp_name in enumerate(SPECIES_COLORS):
        sub = train[train["species"] == sp_name]
        for c_, f in enumerate(NUMERIC):
            ref_s = {g: float(sub[g].median()) for g in NUMERIC}
            ref_s.update({"species": sp_name, "sex": sub["sex"].mode().iloc[0]})
            g = np.linspace(sub[f].min(), sub[f].max(), 100)
            ax = axes[r_, c_]
            for nm, color, ls in (("OLS", C_OLS, "-"), ("M2", C_M2, "-"), ("M3", C_M3, "--")):
                if nm in models:
                    ax.plot(g, models[nm].predict(make_frame(ref_s, f, g)), color=color, ls=ls, lw=2, label=nm)
            ax.plot(sub[f], np.full(len(sub), 0.015), "|", color="k", alpha=0.4, transform=ax.get_xaxis_transform())
            ax.set(title=f"{sp_name} · n = {len(sub)}", xlabel=LABELS[f] if r_ == 2 else "", ylabel="Dự đoán (g)" if c_ == 0 else "")
            ax.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Dự đoán có điều kiện trong miền dữ liệu của từng loài (biến khác ở median của loài; giới tính = mode của loài)")
    savefig(fig, "fig05_conditional_by_species.png")

    # ============================ E. Nhu cầu phi tuyến ============================
    section("E. NHU CẦU THÀNH PHẦN PHI TUYẾN")
    pre = m2_full.named_steps["preprocess"]
    Xnum = pre.named_transformers_["numeric"].named_steps["imputer"].transform(Xtr[NUMERIC])
    cat_idx = [i for i, nm in enumerate(names) if "_sp_" not in nm]
    sp_idx = {f: [i for i, nm in enumerate(names) if nm.startswith(f + "_sp_")] for f in NUMERIC}
    y_ = ytr.to_numpy(float)
    ones = np.ones((len(y_), 1))
    full = np.hstack([ones, Z])
    restricted = np.hstack([ones, Xnum, Z[:, cat_idx]])

    def rss_rank(D):
        beta = np.linalg.lstsq(D, y_, rcond=None)[0]
        r = y_ - D @ beta
        return float(r @ r), int(np.linalg.matrix_rank(D)), r

    # (1) kiểm tra lồng nhau: cột của mô hình tuyến tính phải nằm trong span của mô hình spline
    coef_proj = np.linalg.lstsq(full, restricted, rcond=None)[0]
    nest_err = float(np.abs(full @ coef_proj - restricted).max())
    print(f"Kiểm tra lồng nhau: sai số lớn nhất khi biểu diễn OLS bằng cơ sở spline = {nest_err:.2e} (≈ 0 ⇒ OLS ⊂ M2)")
    assert nest_err < 1e-6

    rss1, p1, res_full = rss_rank(full)
    rss0, p0, _ = rss_rank(restricted)
    n_ = len(y_)
    F = ((rss0 - rss1) / (p1 - p0)) / (rss1 / (n_ - p1))
    pF = float(stats.f.sf(F, p1 - p0, n_ - p1))
    print(f"Tham số: OLS = {p0}, M2 = {p1} (thêm {p1 - p0}) | RSS OLS = {rss0:,.0f}, RSS M2 = {rss1:,.0f}")
    print(f"F-test kinh điển (chung 3 biến): F({p1 - p0}, {n_ - p1}) = {F:.3f}, p = {pF:.4g}")

    # (2) F-test từng biến (3 bậc tự do) + Holm
    part_rows = []
    for f in NUMERIC:
        keep = [i for i in range(Z.shape[1]) if i not in sp_idx[f]]
        j = NUMERIC.index(f)
        Dj = np.hstack([ones, Z[:, keep], Xnum[:, [j]]])
        rssj, pj, _ = rss_rank(Dj)
        Fj = ((rssj - rss1) / (p1 - pj)) / (rss1 / (n_ - p1))
        part_rows.append({"biến": f, "df_tử": p1 - pj, "df_mẫu": n_ - p1, "F": Fj, "p": float(stats.f.sf(Fj, p1 - pj, n_ - p1))})
    order = np.argsort([r["p"] for r in part_rows])
    for rank, i in enumerate(order):
        part_rows[i]["p_Holm"] = min(1.0, max(part_rows[j_]["p"] * (len(order) - r_) for r_, j_ in enumerate(order[: rank + 1])))
    part_tbl = pd.DataFrame(part_rows)
    print(part_tbl.round(4).to_string(index=False))

    # (3) Wald F với HC3 (bền với phương sai không đồng nhất)
    Q0, _ = np.linalg.qr(restricted)
    resid_sp = Z[:, [i for f in NUMERIC for i in sp_idx[f]]]
    resid_sp = resid_sp - Q0 @ (Q0.T @ resid_sp)
    U, s, _ = np.linalg.svd(resid_sp, full_matrices=False)
    r_nl = int((s > 1e-8 * s[0]).sum())
    D = np.hstack([restricted, U[:, :r_nl]])
    beta = np.linalg.lstsq(D, y_, rcond=None)[0]
    e = y_ - D @ beta
    assert abs(e @ e - rss1) / rss1 < 1e-8
    XtXi = np.linalg.inv(D.T @ D)
    h = np.einsum("ij,jk,ik->i", D, XtXi, D)
    w = e / (1 - h)
    V = XtXi @ (D.T @ (D * (w ** 2)[:, None])) @ XtXi
    b, Vb = beta[-r_nl:], V[-r_nl:, -r_nl:]
    Fh = float(b @ np.linalg.solve(Vb, b) / r_nl)
    pFh = float(stats.f.sf(Fh, r_nl, n_ - D.shape[1]))
    print(f"Wald–F bền vững (HC3) cho {r_nl} thành phần phi tuyến: F = {Fh:.3f}, p = {pFh:.4g}")

    aic = lambda rss, k: n_ * np.log(rss / n_) + 2 * k
    bic = lambda rss, k: n_ * np.log(rss / n_) + np.log(n_) * k
    print(f"AIC (trong mẫu): OLS {aic(rss0, p0):.2f} | M2 {aic(rss1, p1):.2f}   BIC: OLS {bic(rss0, p0):.2f} | M2 {bic(rss1, p1):.2f}")

    # (4) Residual bootstrap dưới H0 tuyến tính: biên độ cong của M2 nếu thật sự tuyến tính
    ols_fit = ols.predict(Xtr)
    resid0 = (y_ - ols_fit) * np.sqrt(n_ / (n_ - p0))
    rng_n = np.random.default_rng(SEED)
    obs_amp = {f: amplitude(grids[f], curves["M2"][f], lims_[f][2], lims_[f][3])[0] for f in NUMERIC}
    null_amp = {f: [] for f in NUMERIC}
    for _ in range(N_NULL_BOOT):
        ystar = ols_fit + rng_n.choice(resid0, size=n_)
        mb = build_m2().fit(Xtr, ystar)
        for f in NUMERIC:
            null_amp[f].append(amplitude(grids[f], mb.predict(make_frame(ref, f, grids[f])), lims_[f][2], lims_[f][3])[0])
    null_rows = []
    for f in NUMERIC:
        na = np.array(null_amp[f])
        null_rows.append({"biến": f, "biên độ quan sát (g)": obs_amp[f], "null median (g)": float(np.median(na)),
                          "null p95 (g)": float(np.percentile(na, 95)), "p bootstrap": float((1 + (na >= obs_amp[f]).sum()) / (N_NULL_BOOT + 1))})
    null_tbl = pd.DataFrame(null_rows)
    print("\nResidual bootstrap dưới H0 (1000 lần):")
    print(null_tbl.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), layout="constrained")
    for ax, f in zip(axes, NUMERIC):
        ax.hist(null_amp[f], bins=35, color="#B8C7D6", edgecolor="white")
        ax.axvline(obs_amp[f], color=C_M2, lw=2, label=f"quan sát = {obs_amp[f]:.1f} g")
        ax.set(title=LABELS[f], xlabel="Biên độ lệch khỏi đường thẳng (g)")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Số lần mô phỏng")
    fig.suptitle("Biên độ cong của M2 dưới H0 (dữ liệu giả lập từ mô hình tuyến tính + phần dư bootstrap)")
    savefig(fig, "fig06_nonlinearity_null.png")

    # Kết hợp kết quả E vào một bảng
    e_tbl = pd.DataFrame([
        {"phương pháp": "F-test lồng nhau (9 df, mọi biến)", "thống kê": F, "p": pF},
        {"phương pháp": "Wald–F HC3 (9 df)", "thống kê": Fh, "p": pFh},
        *[{"phương pháp": f"F-test từng biến (3 df): {r['biến']}", "thống kê": r["F"], "p": r["p"]} for r in part_rows],
    ])
    e_tbl.to_csv(TAB / "E_nonlinearity_tests.csv", index=False)
    part_tbl.to_csv(TAB / "E_partial_ftests.csv", index=False)
    null_tbl.to_csv(TAB / "E_residual_bootstrap_null.csv", index=False)
    RESULTS["E"] = {"nest_err": nest_err, "params_ols": p0, "params_m2": p1, "rss_ols": rss0, "rss_m2": rss1, "F": F, "p_F": pF,
                    "F_HC3": Fh, "p_HC3": pFh, "partial": part_rows, "null": null_rows,
                    "aic": {"ols": aic(rss0, p0), "m2": aic(rss1, p1)}, "bic": {"ols": bic(rss0, p0), "m2": bic(rss1, p1)},
                    "shapiro_train_resid_full": dict(zip(("W", "p"), map(float, stats.shapiro(res_full))))}
    print("Shapiro (phần dư M2 trên train):", RESULTS["E"]["shapiro_train_resid_full"])

    (OUT / "results.json").write_text(json.dumps(RESULTS, default=jsonable, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nĐã ghi kết quả vào {OUT.relative_to(OUT.parents[1])}")


if __name__ == "__main__":
    main()
