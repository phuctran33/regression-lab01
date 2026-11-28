"""Figures and text equivalents for the notebook prediction panel."""
from html import escape
from io import BytesIO

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from .modeling import CATEGORICAL, TARGET

STYLES = {
    "OLS": ("#0d6870", "-", "o"), "M1": ("#845126", "--", "s"),
    "M2": ("#805184", ":", "^"), "M3": ("#395b94", "-.", "D"),
    "M4": ("#75473e", "--", "P"),
}


def number(value, digits=1):
    return f"{value:,.{digits}f}".translate(str.maketrans(",.", ".,"))


def table(headers, rows):
    heading = "".join(f'<th scope="col">{escape(str(h))}</th>' for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in row) + "</tr>" for row in rows)
    return f"<details><summary>Xem số liệu của biểu đồ</summary><table><thead><tr>{heading}</tr></thead><tbody>{body}</tbody></table></details>"


def finish(fig):
    for ax in fig.axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#70888c")
        ax.tick_params(colors="#334f57", labelsize=10)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#dce6e3", linewidth=.7)
    with BytesIO() as stream:
        fig.savefig(stream, format="png", dpi=135, facecolor="white")
        return stream.getvalue()


def trend_plot(selected, feature, labels, point, values, estimate, trends, support, radius):
    categorical = feature in CATEGORICAL
    difference = not categorical and {"OLS", "M3"}.issubset(trends)
    fig = Figure(figsize=(6.3, 4.3 if difference else 3.2), layout="constrained")
    if difference:
        ax, delta = fig.subplots(2, 1, sharex=True, height_ratios=(3, 1))
    else:
        ax = fig.subplots()
    sample = support.dropna(subset=[feature])
    current = point[feature].iloc[0]
    x = np.arange(len(values)) if categorical else values
    selected_x = list(values).index(current) if categorical else current
    measured = sample[feature].map(dict(zip(values, x))) if categorical else sample[feature]
    ax.scatter(measured, sample[TARGET], s=13, color="#879da0", alpha=.60, linewidth=0, label="Quan sát train")
    for i, (name, y) in enumerate(trends.items()):
        color, line, marker = STYLES.get(name, ("#3d5160", "--", "s"))
        if categorical:
            offset = (i - (len(trends) - 1) / 2) * .10
            ax.scatter(x + offset, y, color=color, marker=marker, s=42, label=name)
        else:
            ax.plot(x, y, color=color, linestyle=line, lw=2, label=name)
    if radius is not None:
        ax.errorbar(selected_x, estimate[selected], yerr=radius, color="#263d4e", capsize=5,
                    lw=1.7, fmt="none", label="Khoảng cá thể")
    ax.scatter(selected_x, estimate[selected], marker="*", s=140, color="#b66108",
               edgecolor="#412b1b", linewidth=.7, zorder=6, label="Đang chọn")
    if categorical:
        ax.set_xticks(x, [{"female": "Cái", "male": "Đực"}.get(v, v) for v in values])
    else:
        ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.set_ylabel("Khối lượng (g)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, frameon=False,
              fontsize=8.5, handlelength=2, columnspacing=1.2, borderaxespad=0)
    if difference:
        delta.axhline(0, color="#556a70", lw=.8, ls=":")
        delta.plot(x, trends["M3"] - trends["OLS"], color=STYLES["M3"][0])
        delta.set_ylabel("M3 − OLS (g)", fontsize=9)
        delta.yaxis.set_major_locator(MaxNLocator(3))
    (delta if difference else ax).set_xlabel(labels.get(feature, feature))
    # Sample the numeric curve at three actual plotted locations. Categories are all shown.
    indices = range(len(values)) if categorical else [0, len(values) // 2, len(values) - 1]
    rows = [[values[i] if categorical else number(values[i]), *[number(y[i]) for y in trends.values()]]
            for i in indices]
    label = labels.get(feature, feature)
    alt = f"Dự đoán theo {label}. Dấu sao ứng với {selected}, {number(estimate[selected])} g."
    caption = "Mỗi đường hoặc nhóm ký hiệu chỉ thay một đầu vào. Các chấm xám là cá thể train với nhiều tổ hợp số đo."
    if not categorical:
        first, last = trends[selected][0], trends[selected][-1]
        caption += (f" Ở hai đầu miền đo, {escape(selected)} lần lượt dự đoán {number(first)} g và {number(last)} g.")
    if difference:
        caption += " Ô dưới thể hiện chênh lệch M3 trừ OLS theo gram."
    return finish(fig), alt, caption + table([label, *[f"{n} (g)" for n in trends]], rows)


def error_plot(view, name, residuals, species):
    fig = Figure(figsize=(6.3, 3.2), layout="constrained")
    ax = fig.subplots()
    if view == "hist":
        counts, bins, _ = ax.hist(residuals, bins="sturges", color="#36848a", edgecolor="white")
        ax.axvline(0, color="#5f3219", ls="--", lw=1.8, label="Sai số bằng 0")
        ax.legend(frameon=False, fontsize=9)
        ax.set(xlabel="Thực tế − dự đoán (g)", ylabel="Số cá thể")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        alt = f"Histogram phần dư {name}, {len(residuals)} cá thể test. Trung vị {number(np.median(residuals))} g."
        caption = alt + " Phần dư dương nghĩa là mô hình ước lượng thấp hơn cân nặng thực."
        rows = [(number(lo), number(hi), int(n)) for lo, hi, n in zip(bins[:-1], bins[1:], counts)]
        caption += table(["Từ (g)", "Đến (g)", "Số cá thể"], rows)
    else:
        groups = residuals.abs().groupby(species.fillna("Thiếu"), sort=True)
        names, data = zip(*groups)
        ax.boxplot(data, orientation="horizontal", tick_labels=names, patch_artist=True, widths=.5,
                   boxprops={"facecolor": "#b2d9ce", "edgecolor": "#22575d"},
                   medianprops={"color": "#103e46", "linewidth": 2.3},
                   flierprops={"marker": "o", "markersize": 4, "markeredgecolor": "#704218"})
        ax.set(xlabel="Sai số tuyệt đối (g)", ylabel="Loài")
        ax.set_xlim(left=0)
        rows = [(n, len(group), *[number(v) for v in np.percentile(group, [25, 50, 75])])
                for n, group in zip(names, data)]
        alt = f"Sai số tuyệt đối của {name} theo loài trên {len(residuals)} cá thể test."
        caption = alt + " Hộp chứa 50% giá trị giữa. Vạch trong hộp là trung vị. Số cá thể mỗi loài khác nhau."
        caption += table(["Loài", "n", "Q1 (g)", "Trung vị (g)", "Q3 (g)"], rows)
    return finish(fig), alt, caption
