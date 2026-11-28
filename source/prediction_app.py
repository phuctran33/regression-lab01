"""Notebook controls for exploring predictions from already fitted pipelines."""

from html import escape
from io import BytesIO

import ipywidgets as widgets
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .evaluation import fit_ols_inference, ols_prediction_intervals
from .modeling import CATEGORICAL, FEATURES, TARGET

MODEL_STYLES = {
    "OLS": {"color": "#a54c20", "linestyle": "--", "marker": "s"},
    "M1": {"color": "#5b3a83", "linestyle": "-.", "marker": "^"},
    "M2": {"color": "#19754c", "linestyle": ":", "marker": "o"},
    "M3": {"color": "#156581", "linestyle": "-", "marker": "D"},
    "M4": {"color": "#871f52", "linestyle": (0, (5, 2, 1, 2, 1, 2)), "marker": "P"},
}


class PredictionApp:
    def __init__(self, models, train, labels):
        self.models = models
        self.train = train
        self.labels = labels
        self.numeric_features = [feature for feature in FEATURES if feature not in CATEGORICAL]
        self.ols_inference = fit_ols_inference(models["OLS"], train)[0] if "OLS" in models else None
        style = {"description_width": "initial"}
        self.model_picker = widgets.Dropdown(
            options=list(models), description="Mô hình", style=style,
        )
        self.feature_picker = widgets.Dropdown(
            options=[(labels.get(feature, feature), feature) for feature in FEATURES],
            description="Biến trên biểu đồ", style=style,
        )
        can_compare = len(models) > 1
        self.compare = widgets.Checkbox(
            value=can_compare, disabled=not can_compare, description="So sánh các mô hình sẵn sàng",
            indent=False, layout=widgets.Layout(width="auto"),
        )
        self.predict_button = widgets.Button(
            description="Dự đoán", button_style="primary", disabled=not models,
            layout=widgets.Layout(height="44px", margin="6px 0"),
        )
        self.result = widgets.HTML()
        self.chart = widgets.Image(format="png", layout=widgets.Layout(width="100%", max_width="860px"))
        self.controls = {}
        for feature in FEATURES:
            observed = train[feature].dropna()
            label = labels.get(feature, feature)
            if feature in CATEGORICAL:
                control = widgets.Dropdown(
                    options=sorted(observed.unique()), value=observed.mode().iloc[0],
                    description=label, style=style,
                )
            else:
                slider = widgets.IntSlider if feature == "year" else widgets.FloatSlider
                control = slider(
                    value=observed.median(), min=observed.min(), max=observed.max(),
                    step=1 if feature == "year" else 0.1, continuous_update=False,
                    description=label, style=style,
                )
            control.layout.width = "min(100%, 440px)"
            self.controls[feature] = control

        for control in (self.model_picker, self.feature_picker, self.compare, *self.controls.values()):
            control.observe(self._input_changed, names="value")
        self.predict_button.on_click(self._render)
        self.panel = widgets.VBox(
            [self.model_picker, *self.controls.values(), self.feature_picker, self.compare,
             self.predict_button, self.result, self.chart],
        )
        self._input_changed()

    def _message(self, message):
        self.result.value = f'<div role="status" aria-live="polite">{message}</div>'

    def _input_changed(self, _change=None):
        self.chart.value = b""
        if self.models:
            self._message("Bấm Dự đoán để cập nhật kết quả và biểu đồ.")
        else:
            self._message("Chưa có mô hình sẵn sàng. Hãy chạy cell huấn luyện.")

    def _render(self, _button=None):
        self.predict_button.disabled = True
        try:
            name = self.model_picker.value
            feature = self.feature_picker.value
            if name not in self.models or feature not in self.controls:
                raise ValueError("Model or feature is unavailable.")
            inputs = {key: widget.value for key, widget in self.controls.items()}
            if any(inputs[key] not in self.controls[key].options for key in FEATURES if key in CATEGORICAL):
                raise ValueError("A categorical input is missing or unavailable.")
            point = pd.DataFrame([inputs])
            if not np.isfinite(point[self.numeric_features].to_numpy(dtype=float)).all():
                raise ValueError("A numeric input is not finite.")
            names = [name]
            if self.compare.value:
                names.extend(model for model in self.models if model != name)
            focus = self.controls[feature]
            categorical = feature in CATEGORICAL
            if categorical:
                values = list(focus.options)
            elif feature == "year":
                values = np.arange(focus.min, focus.max + 1)
            else:
                values = np.linspace(focus.min, focus.max, 100)
            scenarios = point.loc[point.index.repeat(len(values))].assign(**{feature: values})
            estimate = {model: self.models[model].predict(point)[0] for model in names}
            trends = {model: self.models[model].predict(scenarios) for model in names}
            if not np.isfinite(list(estimate.values())).all() or not np.isfinite(list(trends.values())).all():
                raise ValueError("The model returned a non-finite prediction.")
            selected = f'<strong>{escape(name)} · body_mass_g = {estimate[name]:,.1f} g</strong>'
            alternatives = [f"{escape(model)} {estimate[model]:,.1f} g" for model in names[1:]]
            messages = [" · ".join([selected, *alternatives])]
            if name == "OLS":
                intervals = ols_prediction_intervals(self.models[name], self.ols_inference, point).iloc[0]
                messages.append(
                    f"OLS cổ điển · CI 95% cho trung bình [{intervals.mean_ci_lower:,.1f} đến "
                    f"{intervals.mean_ci_upper:,.1f}] g · PI 95% cho cá thể [{intervals.obs_ci_lower:,.1f} đến "
                    f"{intervals.obs_ci_upper:,.1f}] g"
                )
            species = inputs["species"]
            measured = self.train.loc[self.train["species"].eq(species), self.numeric_features]
            outside = [
                escape(self.labels.get(key, key)) for key in self.numeric_features
                if inputs[key] < measured[key].min() or inputs[key] > measured[key].max()
            ]
            if outside:
                messages.append(f"Ngoài khoảng train của loài {escape(species)} · {', '.join(outside)}.")
            self.chart.value = self._plot(name, feature, point, values, estimate, trends)
            self._message("<br>".join(messages))
        except ValueError:
            self.chart.value = b""
            self._message("Không thể dự đoán. Hãy chọn mô hình và kiểm tra các giá trị đầu vào.")
        finally:
            self.predict_button.disabled = not self.models

    def _plot(self, selected, feature, point, values, estimate, trends):
        categorical = feature in CATEGORICAL
        show_difference = not categorical and {"OLS", "M3"}.issubset(trends)
        fig = Figure(figsize=(8, 5.0 if show_difference else 3.8), layout="constrained")
        if show_difference:
            ax, difference = fig.subplots(2, 1, sharex=True, height_ratios=(3, 1))
        else:
            ax = fig.subplots()
        observed = self.train.dropna(subset=[feature])
        x = values
        observed_x = observed[feature]
        selected_x = point[feature].iloc[0]
        if categorical:
            positions = {value: index for index, value in enumerate(values)}
            x = np.arange(len(values))
            observed_x = observed[feature].map(positions)
            selected_x = positions[selected_x]
            ax.set_xticks(x, values)
        ax.scatter(observed_x, observed[TARGET], s=15, alpha=0.28, color="#526478",
                   label="Quan sát train")
        for index, (name, trend) in enumerate(trends.items()):
            style = MODEL_STYLES.get(name, {"color": "#5b3a83", "linestyle": "-.", "marker": "^"})
            if categorical:
                offset = (index - (len(trends) - 1) / 2) * 0.12
                ax.scatter(x + offset, trend, label=name, color=style["color"], marker=style["marker"])
            else:
                ax.plot(x, trend, label=name, color=style["color"], linestyle=style["linestyle"], lw=2)
        ax.scatter(selected_x, estimate[selected], marker="*", s=135, color="#202938", zorder=5,
                   edgecolor="white", label=f"Đầu vào đã chọn · {selected}")
        ax.set(ylabel=self.labels[TARGET], title="Dự đoán khi giữ cố định các đầu vào còn lại")
        ax.legend(frameon=False, loc="best")
        ax.grid(axis="y", alpha=0.18)
        x_label = self.labels.get(feature, feature)
        if show_difference:
            difference.plot(x, trends["M3"] - trends["OLS"], color=MODEL_STYLES["M3"]["color"])
            difference.axhline(0, color="#526478", linestyle=":", linewidth=1)
            difference.set(xlabel=x_label, ylabel="M3 − OLS (g)")
        else:
            ax.set_xlabel(x_label)
        with BytesIO() as buffer:
            fig.savefig(buffer, format="png", dpi=120)
            return buffer.getvalue()
