"""Figures and text equivalents for the notebook prediction panel."""
from html import escape
from io import BytesIO
from typing import NamedTuple

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.ticker import FuncFormatter, MaxNLocator

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


class Mark(NamedTuple):
    x: float
    y: float
    title: str
    detail: str
    axis: int = 0


def finish(fig, marks, *, categorical_x=False, categorical_y=False, selected=0):
    """Draw the same artists on wide and narrow canvases, retaining both sets of labels."""
    def tick(value, _position):
        return number(value, 0 if float(value).is_integer() else 1).replace("-", "−")

    for ax in fig.axes:
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#a0b4c3")
        ax.tick_params(colors="#384f65", labelsize=9, length=0, pad=5)
        ax.set_axisbelow(True)
        ax.grid(axis="x" if categorical_y else "y", color="#d8e3eb", linewidth=.7)
        if not categorical_y:
            ax.yaxis.set_major_formatter(FuncFormatter(tick))
        if not categorical_x:
            ax.xaxis.set_major_formatter(FuncFormatter(tick))

    fig.set_dpi(145)
    canvas = FigureCanvasAgg(fig)
    def capture():
        # Project with the transforms of the exact canvas the browser displays.
        with BytesIO() as stream:
            canvas.print_png(stream)
            pixels = stream.getvalue()
        width, height = fig.bbox.size
        coords = []
        for m in marks:
            x, y = fig.axes[m.axis].transData.transform((m.x, m.y))
            coords.append([round(x/width, 5), round(1-y/height, 5)])
        return pixels, coords

    wide, wide_coords = capture()

    fig.set_size_inches(3.05, 4.55 if len(fig.axes) > 1 else 3.25)
    for ax in fig.axes:
        if not categorical_x:
            ax.xaxis.set_major_locator(MaxNLocator(4))
        legend = ax.get_legend()
        if legend is not None and len(legend.get_texts()) > 1:
            ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02, 1, .1), mode="expand",
                      ncols=2, frameon=False, fontsize=8, handlelength=1.3,
                      handletextpad=.35, borderaxespad=0, columnspacing=.7)
        if ax.get_aspect() == 1:
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.yaxis.set_major_locator(MaxNLocator(4))
    compact, compact_coords = capture()
    return wide, compact, dict(labels=[[m.title, m.detail] for m in marks], wide=wide_coords,
                              compact=compact_coords, selected=int(selected))


def trend_plot(selected, feature, labels, point, values, estimate, trends, support, radius):
    categorical = feature in CATEGORICAL
    difference = not categorical and {"OLS", "M3"}.issubset(trends)
    fig = Figure(figsize=(7.2, 4.45 if difference else 3.2), layout="constrained")
    if difference:
        ax, delta = fig.subplots(2, 1, sharex=True, height_ratios=(3, 1))
    else:
        ax = fig.subplots()
    sample = support.dropna(subset=[feature])
    current = point[feature].iloc[0]
    x = np.arange(len(values)) if categorical else values
    selected_x = list(values).index(current) if categorical else current
    measured = sample[feature].map(dict(zip(values, x))) if categorical else sample[feature]
    ax.scatter(measured, sample[TARGET], s=13, color="#617a90", alpha=.90, linewidth=0, label="Quan sát train")
    marks = [Mark(selected_x, estimate[selected], f"{selected} · cá thể đang khảo sát",
                  f"Dự đoán {number(estimate[selected])} g" + (
                      f" · Khoảng {number(estimate[selected]-radius)}–{number(estimate[selected]+radius)} g" if radius is not None else ""))]
    feature_label = labels.get(feature, feature)
    for i, (name, y) in enumerate(trends.items()):
        color, line, marker = STYLES.get(name, ("#3d5160", "--", "s"))
        if categorical:
            offset = (i - (len(trends) - 1) / 2) * .10
            ax.scatter(x + offset, y, color=color, marker=marker, s=42, label=name)
        else:
            ax.plot(x, y, color=color, linestyle=line, lw=2, label=name)
        for j, value in enumerate(y):
            v = str(values[j]) if categorical else number(values[j])
            marks.append(Mark(x[j] + (offset if categorical else 0), value, f"{name} · xu hướng",
                              f"{feature_label} {v} · Dự đoán {number(value)} g"))
    for idx, xf in measured.items():
        row = sample.loc[idx]
        marks.append(Mark(xf, row[TARGET], f"Train · ID {idx} · {row['species']}",
                          f"{feature_label} {row[feature]} · Thực tế {number(row[TARGET])} g"))
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
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=4, frameon=False,
              fontsize=8.5, handlelength=2, columnspacing=1.2, borderaxespad=0)
    if difference:
        delta.axhline(0, color="#556a70", lw=.8, ls=":")
        delta.plot(x, trends["M3"] - trends["OLS"], color=STYLES["M3"][0])
        for xf, delta_g in zip(x, trends["M3"] - trends["OLS"]):
            marks.append(Mark(xf, delta_g, "M3 − OLS", f"{feature_label} {number(xf)} · Chênh lệch {number(delta_g)} g", 1))
        delta.set_ylabel("M3 − OLS (g)", fontsize=9)
        delta.yaxis.set_major_locator(MaxNLocator(3))
    (delta if difference else ax).set_xlabel(labels.get(feature, feature))
    # Sample the numeric curve at three actual plotted locations. Categories are all shown.
    indices = range(len(values)) if categorical else [0, len(values) // 2, len(values) - 1]
    rows = [[{"female": "Cái", "male": "Đực"}.get(values[i], values[i]) if categorical else number(values[i]), *[number(y[i]) for y in trends.values()]]
            for i in indices]
    label = labels.get(feature, feature)
    alt = f"Dự đoán theo {label}. Dấu sao ứng với {selected}, {number(estimate[selected])} g."
    caption = "Mỗi đường hoặc nhóm ký hiệu chỉ thay một đầu vào. Các chấm xám là cá thể train với nhiều tổ hợp số đo."
    if not categorical:
        first, last = trends[selected][0], trends[selected][-1]
        caption += (f" Ở hai đầu miền đo, {escape(selected)} lần lượt dự đoán {number(first)} g và {number(last)} g.")
    if difference:
        caption += " Ô dưới thể hiện chênh lệch M3 trừ OLS theo gram."
    return finish(fig, marks, categorical_x=categorical), alt, caption + table([label, *[f"{n} (g)" for n in trends]], rows)


def error_plot(view, name, residuals, species):
    fig = Figure(figsize=(7.2, 3.2), layout="constrained")
    ax = fig.subplots()
    if view == "hist":
        counts, bins, _ = ax.hist(residuals, bins="sturges", color="#36848a", edgecolor="white")
        ax.axvline(0, color="#5f3219", ls="--", lw=1.8, label="Sai số bằng 0")
        ax.legend(frameon=False, fontsize=9)
        ax.set(xlabel="Thực tế − dự đoán (g)", ylabel="Số cá thể")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        alt = f"Histogram phần dư {name}, {len(residuals)} cá thể test. Trung vị {number(np.median(residuals))} g."
        median = np.median(residuals)
        direction = "thấp" if median > 0 else "cao"
        bias = f" Dấu của trung vị cho thấy xu hướng dự đoán {direction} trên tập này." if median else " Trung vị phần dư bằng 0."
        caption = alt + bias + " Phần dư dương nghĩa là mô hình ước lượng thấp hơn cân nặng thực."
        rows = [(number(lo), number(hi), int(n)) for lo, hi, n in zip(bins[:-1], bins[1:], counts)]
        marks = [Mark((lo+hi)/2, n, f"{name} · {int(n)} cá thể",
                      f"Phần dư từ {number(lo)} đến {number(hi)} g") for lo, hi, n in zip(bins[:-1], bins[1:], counts)]
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
        medians = groups.median()
        caption += (f" Trung vị sai số lớn nhất ở {escape(str(medians.idxmax()))} ({number(medians.max())} g), "
                    f"nhỏ nhất ở {escape(str(medians.idxmin()))} ({number(medians.min())} g).")
        caption += table(["Loài", "n", "Q1 (g)", "Trung vị (g)", "Q3 (g)"], rows)
        marks = [Mark(group.median(), i, f"{name} · {species_name} · {len(group)} cá thể",
                      f"Q1 {number(group.quantile(.25))} g · Trung vị {number(group.median())} g · Q3 {number(group.quantile(.75))} g")
                 for i, (species_name, group) in enumerate(zip(names, data), 1)]
    return finish(fig, marks, categorical_y=view == "box"), alt, caption
