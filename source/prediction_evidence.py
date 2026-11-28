"""Read-only views of fitted predictors and the published cross-validation experiment."""
import json
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .analysis.uncertainty import LEVELS
from .modeling import CATEGORICAL, FEATURES, TARGET
from .prediction_figures import Mark, STYLES, finish, number, table

ARTIFACT = Path(__file__).resolve().parents[1] / "report/comparison.json"


def demo_rows(train):
    complete = train.dropna(subset=[*FEATURES, TARGET])
    numeric = [f for f in FEATURES if f not in CATEGORICAL]
    result = {}
    for species, group in complete.groupby("species"):
        scales = group[numeric].std().replace(0, 1).fillna(1)
        distance = ((group[numeric] - group[numeric].median()) / scales).pow(2).sum(axis=1)
        result[f"{species} · gần trung vị"] = distance.idxmin()
    if len(complete):
        result["Cá thể nhẹ nhất train có đủ số đo"] = complete[TARGET].idxmin()
        result["Cánh bơi dài nhất train có đủ số đo"] = complete["flipper_length_mm"].idxmax()
    return result


def test_rows(test, prediction, radius=None):
    frame = pd.DataFrame({"actual": test[TARGET], "pred": prediction}, index=test.index)
    frame["species"] = test.get("species", "")
    frame["residual"] = frame.actual - frame.pred
    if radius is not None:
        frame["low"], frame["high"] = frame.pred - radius, frame.pred + radius
        frame["inside"] = frame.actual.between(frame.low, frame.high)
    return frame.sort_values("pred", kind="stable")


def comparison_rows(calibrated, performance, point, level):
    rows = []
    for name, model in calibrated.items():
        pred, lo, hi = (float(a[0]) for a in model.interval(point, level))
        summary = performance[name, level]
        rows.append(dict(model=name, pred=pred, low=lo, high=hi, width=hi-lo,
                         coverage=summary["coverage"], mae=summary["MAE (g)"]))
    return pd.DataFrame(rows).set_index("model")


def verified_benchmark(test, predictions, path=ARTIFACT):
    if not path.is_file() or test is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    recorded = pd.DataFrame(data["test_predictions"]).set_index("row_id")
    if not recorded.index.equals(test.index) or not predictions.keys() <= set(recorded.columns):
        return None
    for name, values in predictions.items():
        if not np.allclose(recorded[name], values, rtol=0, atol=1e-6):
            return None
        actual_mae = np.mean(np.abs(test[TARGET] - values))
        if not np.isclose(actual_mae, data["benchmark"][name]["MAE (g)"], rtol=0, atol=1e-6):
            return None
    return data


def canvas():
    fig = Figure(figsize=(7.2, 3.35), layout="constrained")
    return fig, fig.subplots()


def legend(ax, columns=2):
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncols=columns,
              frameon=False, fontsize=8.5, borderaxespad=0, handlelength=1.6)


def interval_plot(rows, level, selected):
    fig, ax = canvas()
    for y, (name, row) in enumerate(rows.iterrows()):
        color, _, marker = STYLES.get(name, ("#35577b", "-", "o"))
        ax.hlines(y, row.low, row.high, color=color, lw=3)
        ax.plot([row.low, row.high], [y, y], "|", color=color, ms=11)
        ax.plot(row.pred, y, marker=marker, color=color, ms=8)
        if name == selected:
            ax.plot(row.pred, y, "o", mfc="none", mec="#163b39", ms=13)
    ax.set_yticks(range(len(rows)), rows.index)
    ax.invert_yaxis()
    ax.margins(y=.26)
    ax.set_xlabel("Khối lượng cá thể (g)")
    name = rows.width.idxmin()
    alt = (f"Conformal {level:.0%}. {name} có khoảng hẹp nhất ({number(rows.width.min())} g). "
           f"Các dự đoán điểm cách nhau tối đa {number(rows.pred.max() - rows.pred.min())} g.")
    note = " Ký hiệu là dự đoán điểm, hai đầu là cận. Khoảng hẹp có thể bỏ sót nhiều cá thể hơn. Chênh lệch độ rộng chưa đủ để xếp hạng mô hình. Chọn chế độ Benchmark để đối chiếu CV và bootstrap."
    data = [[n, number(r.pred), number(r.low), number(r.high), number(r.width),
             number(100*r.coverage)+"%", number(r.mae)] for n, r in rows.iterrows()]
    marks = [Mark(r.pred, i, f"{n} · mục tiêu {level:.0%}", f"{number(r.pred)} g · Khoảng {number(r.low)}–{number(r.high)} g · Rộng {number(r.width)} g · Bao phủ {number(100*r.coverage)}% · MAE {number(r.mae)} g")
             for i, (n, r) in enumerate(rows.iterrows())]
    return finish(fig, marks, categorical_y=True, selected=rows.index.get_loc(selected)), alt, alt + note + table(
        ["Mô hình", "Điểm (g)", "Cận dưới (g)", "Cận trên (g)", "Rộng (g)", "Bao phủ test", "MAE (g)"], data)


def coverage_plot(names, performance, selected):
    fig, ax = canvas()
    data = []
    marks = []
    for level, style in zip(LEVELS, (":", "--")):
        ax.axhline(100*level, color="#657585", lw=1, ls=style)
    for name in names:
        color, line, marker = STYLES.get(name, ("#35577b", "-", "o"))
        summaries = [performance[name, level] for level in LEVELS]
        ax.plot([s["width_g"] for s in summaries], [100*s["coverage"] for s in summaries],
                color=color, ls=line, marker=marker, ms=7 if name == selected else 5, lw=2 if name == selected else 1, label=name)
        for level, s in zip(LEVELS, summaries):
            marks.append(Mark(s["width_g"], 100*s["coverage"], f"{name} · mục tiêu {level:.0%}",
                              f"Bao phủ {number(100*s['coverage'])}% · Rộng {number(s['width_g'])} g · Ngoài {s['outside']}/{s['n']}"))
            data.append([name, f"{level:.0%}", number(100*s["coverage"])+"%", number(s["width_g"]),
                         s["outside"], s["n"]])
    ax.set(xlabel="Độ rộng khoảng (g)", ylabel="Bao phủ thực nghiệm (%)")
    legend(ax, 4)
    low, high = [performance[selected, level] for level in LEVELS]
    alt = (f"{selected}: từ 90% lên 95%, khoảng rộng thêm {number(high['width_g']-low['width_g'])} g. "
           f"Số cá thể ngoài khoảng lần lượt là {low['outside']} và {high['outside']} trên {high['n']} mẫu.")
    note = (" Mỗi đường nối hai mức của một mô hình, từ khoảng hẹp đến khoảng rộng. "
            "Hai đường ngang đánh dấu mục tiêu 90% và 95%. Bao phủ thực nghiệm là tỷ lệ của tập test.")
    return finish(fig, marks, selected=list(names).index(selected)*len(LEVELS)), alt, alt + note + table(["Mô hình", "Mục tiêu", "Bao phủ test", "Rộng (g)", "Ngoài", "n"], data)


def heldout_plot(rows, name, level):
    fig, ax = canvas()
    x = np.arange(1, len(rows)+1)
    inside = rows.inside.to_numpy()
    ax.fill_between(x, rows.low, rows.high, color="#c4e1df", label="Khoảng cá thể")
    ax.plot(x, rows.pred, color="#194964", lw=1.6, label="Dự đoán")
    ax.scatter(x[inside], rows.actual[inside], s=13, facecolor="white", edgecolor="#245c65", label="Trong khoảng")
    ax.scatter(x[~inside], rows.actual[~inside], s=40, color="#994015", marker="x", label="Ngoài khoảng")
    ax.set(xlabel="Thứ tự theo dự đoán tăng dần", ylabel="Khối lượng (g)")
    legend(ax, 4)
    alt = (f"{name} · mục tiêu {level:.0%}. {inside.sum()}/{len(rows)} cá thể trong khoảng "
           f"({number(100*inside.mean())}%), {sum(~inside)} ngoài khoảng. "
           f"Độ rộng {number((rows.high-rows.low).mean())} g.")
    data = [[i, number(r.actual), number(r.pred), number(r.low), number(r.high),
             "Trong" if r.inside else "Ngoài"] for i, r in rows.iterrows()]
    note = " Dấu × đánh dấu cá thể ngoài khoảng. Thứ tự trên trục ngang và trong bảng xếp theo dự đoán, từ nhẹ đến nặng."
    marks = [Mark(j, r.actual, f"{name} · ID {idx} · {r.species}", f"Thực {number(r.actual)} g · Dự đoán {number(r.pred)} g · Phần dư {number(r.residual)} g · |Sai số| {number(abs(r.residual))} g · Khoảng {level:.0%} {number(r.low)}–{number(r.high)} g · Rộng {number(r.high-r.low)} g · {'Trong' if r.inside else 'Ngoài'} khoảng")
             for j, (idx, r) in enumerate(rows.iterrows(), 1)]
    return finish(fig, marks), alt, alt + note + table(["ID", "Thực (g)", "Dự đoán (g)", "Dưới (g)", "Trên (g)", "Vị trí"], data)


def actual_plot(rows, name, metrics, protocol):
    fig, ax = canvas()
    lo = min(rows.actual.min(), rows.pred.min())
    hi = max(rows.actual.max(), rows.pred.max())
    pad = max(1, (hi-lo)*.04)
    ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], "--", color="#936037", label="y = x")
    ax.scatter(rows.actual, rows.pred, color="#23646a", s=18, alpha=.8, label=name)
    ax.set(xlim=(lo-pad, hi+pad), ylim=(lo-pad, hi+pad), xlabel="Khối lượng thực tế (g)", ylabel="Khối lượng dự đoán (g)")
    ax.set_aspect("equal", adjustable="box")
    legend(ax)
    worst = rows.residual.abs().idxmax()
    alt = (f"{name} · {protocol}, {len(rows)} cá thể. MAE {number(metrics['MAE (g)'])} g, "
           f"RMSE {number(metrics['RMSE (g)'])} g, R² {number(metrics['R²'], 3)}. "
           f"Sai lệch lớn nhất {number(abs(rows.at[worst, 'residual']))} g ở ID {worst}.")
    data = [[i, number(r.actual), number(r.pred), number(r.residual)] for i, r in rows.iterrows()]
    marks = [Mark(r.actual, r.pred, f"{name} · ID {idx} · {r.species}",
                  f"Thực {number(r.actual)} g · Dự đoán {number(r.pred)} g · Phần dư {number(r.residual)} g · |Sai số| {number(abs(r.residual))} g")
             for idx, r in rows.iterrows()]
    return finish(fig, marks), alt, alt + " Điểm dưới đường chéo là dự đoán thấp. MAE tóm tắt trung bình trên cả tập." + table(
        ["ID", "Thực (g)", "Dự đoán (g)", "Phần dư (g)"], data)


def benchmark_plot(data, names, n):
    fig, ax = canvas()
    names = list(names)
    contrasts = [(name, data["paired_delta_mae"][name]) for name in names if name in data["paired_delta_mae"]]
    ax.axvline(0, color="#384f65", lw=1.3, ls="--")
    for i, (name, c) in enumerate(contrasts):
        ax.hlines(i, c["ci_low_g"], c["ci_high_g"], color="#23646a", lw=3)
        ax.plot(c["delta_g"], i, "D", color="#214568")
    ax.set_yticks(range(len(contrasts)), [name for name, _ in contrasts])
    ax.invert_yaxis()
    ax.margins(y=.3)
    ax.set_xlabel("ΔMAE so với OLS (g)")
    folds = pd.DataFrame(data["cv_folds"])
    headers = ["Mô hình", "MAE (g)", "MSE (g²)", "RMSE (g)", "R²", "CV cố định MAE (g)", "Nested CV MAE (g)"]
    rows = []
    for name in names:
        m = data["benchmark"][name]
        means = folds.loc[folds.model.eq(name)].groupby("protocol")["MAE (g)"].mean()
        rows.append([name, number(m["MAE (g)"]), number(m["MSE (g²)"]), number(m["RMSE (g)"]), number(m["R²"], 3),
                     number(means["fixed"]), number(means["nested"])])
    containing = sum(c["ci_low_g"] <= 0 <= c["ci_high_g"] for _, c in contrasts)
    alt = f"Benchmark trên {n} cá thể. {containing}/{len(contrasts)} khoảng bootstrap ghép cặp 95% của ΔMAE chứa 0."
    details = table(headers, rows).replace("Xem số liệu của biểu đồ", "Chỉ số test và MAE trung bình các fold")
    paired = [[name, *[number(c[k], 2) for k in ("delta_g", "ci_low_g", "ci_high_g")]] for name, c in contrasts]
    marks = [Mark(c["delta_g"], i, f"{n} so với OLS", f"ΔMAE {number(c['delta_g'],2)} g · CI 95% [{number(c['ci_low_g'],2)}; {number(c['ci_high_g'],2)}] g")
             for i, (n, c) in enumerate(contrasts)]
    return finish(fig, marks, categorical_y=True), alt, (alt + " Δ âm là giảm MAE. Đọc cả độ lớn theo gram và độ bất định. Chọn chế độ conformal để so khoảng cho cùng một cá thể. "
        "CV cố định giữ cấu hình đã chọn. Nested CV chọn lại trong từng fold train." + details +
        table(["So với OLS", "ΔMAE (g)", "CI dưới (g)", "CI trên (g)"], paired))


def method_plot(scores, name, level, n_train, n_test):
    fig, ax = canvas()
    scores = np.sort(scores)
    n = len(scores)
    k = ceil((n+1)*level)
    ax.plot(np.arange(1, n+1), scores, color="#285772", label="Sai số hiệu chỉnh")
    q = scores[k-1] if k <= n else np.inf
    if np.isfinite(q):
        ax.plot(k, q, "D", color="#994015", label=f"Hạng {k}")
        ax.axhline(q, color="#8b4c27", ls=":", lw=1)
    ax.set(xlabel="Hạng sai số tuyệt đối tăng dần", ylabel="Sai số tuyệt đối (g)")
    legend(ax)
    alt = f"{name} · mục tiêu {level:.0%}, hạng {k}/{n}, bán kính {number(q)} g."
    steps = [
        (f"Học trên {n_train-n} cá thể", f"Tách {n_train} mẫu train trước khi chọn cấu hình và fit."),
        (f"Dự đoán {n} mẫu hiệu chỉnh", "Dùng chính predictor đã fit, giữ riêng nhãn hiệu chỉnh."),
        ("Đo độ lệch", "Lấy trị tuyệt đối của thực tế trừ dự đoán."),
        ("Xếp hạng", f"Sắp {n} sai số từ nhỏ đến lớn."),
        (f"Lấy hạng {k}", f"Làm tròn lên ({n} + 1) × {level:.0%}."),
        ("Lập khoảng cá thể", f"Dự đoán điểm ± {number(q)} g. Độ rộng cố định {number(2*q)} g."),
        (f"Đối chiếu {n_test} mẫu test", "Đếm số khối lượng thực tế nằm giữa hai cận, kể cả bằng cận."),
    ]
    flow = '<ol class="pg-method">' + ''.join(f"<li><strong>{title}</strong> {text}</li>" for title,text in steps) + "</ol>"
    marks = [Mark(i, v, f"{name} · hạng {i}/{n}", f"Sai số hiệu chỉnh {number(v)} g" + (" · Bán kính đã chọn" if i == k else ""))
             for i, v in enumerate(scores, 1)]
    return finish(fig, marks, selected=min(k,n)-1), alt, (alt + flow + "Bao phủ biên áp dụng trên các cá thể có thể trao đổi được với mẫu hiệu chỉnh. "
        "Tỷ lệ có thể khác theo loài. Dữ liệu ít khiến tỷ lệ quan sát dao động. Tập test đã được nhóm đánh giá trước đây. "
        "Đổi predictor cần hiệu chỉnh lại khoảng." + table(["Hạng", "|Phần dư| (g)"], [(i, number(v)) for i,v in enumerate(scores,1)]))
