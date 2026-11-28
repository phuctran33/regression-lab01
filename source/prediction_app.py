"""Notebook interface for exploring individual predictions and held-out errors."""
from base64 import b64encode
from html import escape
from pathlib import Path

import anywidget
import ipywidgets as widgets
import numpy as np
import pandas as pd
import traitlets

from .analysis.uncertainty import LEVELS, MODULES, fit_calibrated, interval_summary, proper_calibration_split
from .evaluation import fit_ols_inference, ols_prediction_intervals
from .modeling import CATEGORICAL, FEATURES, TARGET, regression_metrics
from .prediction_figures import error_plot, number, trend_plot

ASSETS = Path(__file__).with_name("ui")
MODEL_NAMES = {"OLS": "OLS · Tuyến tính", "M1": "M1 · Ridge", "M2": "M2 · Spline cơ sở",
               "M3": "M3 · Spline làm mượt", "M4": "M4 · Huber"}
SEX = {"female": "Cái", "male": "Đực"}


class Measurement(anywidget.AnyWidget):
    _esm = ASSETS / "measurement.js"
    _css = ASSETS / "panel.css"
    description = traitlets.Unicode().tag(sync=True)
    min = traitlets.Float().tag(sync=True)
    max = traitlets.Float().tag(sync=True)
    step = traitlets.Float(.1).tag(sync=True)
    value = traitlets.Float().tag(sync=True)
    valid = traitlets.Bool(True).tag(sync=True)


class Chart(anywidget.AnyWidget):
    _esm = ASSETS / "chart.js"
    value = traitlets.Bytes().tag(sync=True)
    description = traitlets.Unicode().tag(sync=True)


class Result(anywidget.AnyWidget):
    _esm = ASSETS / "result.js"
    value = traitlets.Unicode().tag(sync=True)
    announcement = traitlets.Unicode().tag(sync=True)


class PredictionApp:
    def __init__(self, models, train, labels, *, test=None):
        self.models = models
        self.train = train
        self.test = test
        self.labels = labels
        self.numeric_features = [key for key in FEATURES if key not in CATEGORICAL]
        self.ols_inference = fit_ols_inference(models["OLS"], train)[0] if "OLS" in models else None
        self.calibrated = {}
        self.performance = {}
        self.residuals = {}
        if test is not None:
            if not test.index.intersection(train.index).empty:
                raise ValueError("Test rows overlap training rows.")
            self.proper, calibration = proper_calibration_split(train)
            self.calibrated = fit_calibrated(self.proper, calibration, [n for n in models if n in MODULES])
            for name, fitted in models.items():
                prediction = fitted.predict(test[FEATURES])
                self.residuals["full", name] = test[TARGET] - prediction
                self.performance["full", name] = regression_metrics(test[TARGET], prediction)
            for name, fitted in self.calibrated.items():
                for level in LEVELS:
                    point, low, high = fitted.interval(test[FEATURES], level)
                    self.performance[name, level] = {
                        **regression_metrics(test[TARGET], point),
                        **interval_summary(test[TARGET], low, high),
                    }
                self.residuals["conformal", name] = test[TARGET] - point

        self._revision = 0
        self._rendering = False
        self.rendered = False
        modes = [("Benchmark · toàn bộ train", "full")]
        if self.calibrated:
            modes.insert(0, ("Dự đoán cá thể · conformal", "conformal"))
        self.protocol = widgets.Dropdown(options=modes, description="Chế độ")
        self.level = widgets.Dropdown(options=[("90%", .90), ("95%", .95)], value=.95,
                                      description="Mức bao phủ")
        self.model_picker = widgets.Dropdown(description="Mô hình")
        self.feature_picker = widgets.Dropdown(
            options=[(labels.get(f, f), f) for f in FEATURES], description="Biến trên biểu đồ")
        views = [("Xu hướng có điều kiện", "trend")]
        if self.residuals:
            views.extend([("Phân bố phần dư", "hist"), ("Sai số theo loài", "box")])
        self.view = widgets.Dropdown(options=views, description="Biểu đồ")
        self.compare = widgets.Checkbox(value=len(models) > 1 and not self.calibrated,
                                        description="So sánh các mô hình sẵn sàng", indent=False)
        self.predict_button = widgets.Button(description="Dự đoán")
        self.result = Result()
        self.chart = Chart().add_class("pg-chart")
        self.caption = Result()
        self.controls = {}
        for feature in FEATURES:
            observed = train[feature].dropna()
            label = labels.get(feature, feature)
            if feature in CATEGORICAL:
                options = [(SEX.get(v, v), v) for v in sorted(observed.unique())]
                control = widgets.Dropdown(description=label, options=options, value=observed.mode().iloc[0])
            else:
                control = Measurement(description=label, min=observed.min(), max=observed.max(),
                                      value=observed.median(), step=1 if feature == "year" else .1)
                control.observe(self._input_changed, names="valid")
            self.controls[feature] = control

        art = b64encode((ASSETS / "palmer-penguins.png").read_bytes()).decode("ascii")
        header = widgets.HTML(
            '<header class="pg-heading" lang="vi"><div><h2>Khối lượng chim cánh cụt</h2>'
            '<p>Nhập số đo của một cá thể Palmer Penguins để ước lượng khối lượng và xem khoảng dự đoán.</p></div>'
            f'<figure><img src="data:image/png;base64,{art}" alt="Ba loài Chinstrap, Gentoo và Adelie">'
            '<figcaption><a href="https://allisonhorst.github.io/palmerpenguins/articles/art.html">'
            'Artwork by @allison_horst</a></figcaption></figure></header>')
        form = widgets.VBox([
            widgets.HTML("<h3>Cá thể khảo sát</h3>"),
            widgets.Box([self.controls["species"], self.controls["sex"]]).add_class("pg-pair"),
            *[self.controls[f] for f in self.numeric_features],
            widgets.HTML('<h3 class="pg-settings">Cấu hình dự đoán</h3>'),
            self.protocol, widgets.Box([self.model_picker, self.level]).add_class("pg-pair").add_class("pg-model-pair"),
            self.predict_button,
        ]).add_class("pg-form")
        self.figure_title = widgets.HTML("<h3>Xu hướng khối lượng</h3>")
        self.figure_panel = widgets.VBox([
            self.figure_title,
            widgets.Box([self.view, self.feature_picker]).add_class("pg-pickers"),
            self.compare, self.chart, self.caption,
        ]).add_class("pg-view")
        output = widgets.VBox([self.result, self.figure_panel]).add_class("pg-output")
        self.panel = widgets.VBox([header, widgets.Box([form, output]).add_class("pg-layout")],
                                  layout=widgets.Layout(width="100%", max_width="1140px", margin="0"))
        self.panel.add_class("penguin-panel")
        self.protocol.observe(self._change_protocol, names="value")
        self._change_protocol()
        for control in (self.level, self.model_picker, *self.controls.values()):
            control.observe(self._input_changed, names="value")
        for control in (self.view, self.compare, self.feature_picker):
            control.observe(self._change_view, names="value")
        self.predict_button.on_click(self._render)

    def _change_protocol(self, _change=None):
        conformal = self.protocol.value == "conformal"
        names = self.calibrated if conformal else self.models
        current = self.model_picker.value
        self.model_picker.options = [(MODEL_NAMES.get(n, n), n) for n in names]
        self.model_picker.value = current if current in names else next(iter(names), None)
        self.level.disabled = not conformal
        self.compare.disabled = len(names) < 2
        self._input_changed()

    def _show_notice(self, text):
        self.result.value = f'<p class="pg-notice" lang="vi">{escape(text)}</p>'
        self.result.announcement = text

    def _input_changed(self, _change=None):
        self._revision += 1
        self._rendering = False
        self.predict_button.description = "Dự đoán"
        self.rendered = False
        self.chart.value = b""
        self.caption.value = ""
        self.figure_panel.layout.display = "none"
        invalid = any(not self.controls[key].valid for key in self.numeric_features)
        self.predict_button.disabled = invalid or not self.models
        if not self.models:
            self._show_notice("Chưa có mô hình sẵn sàng. Hãy chạy cell huấn luyện.")
        elif invalid:
            self._show_notice("Kiểm tra số đo được đánh dấu trước khi dự đoán.")
        else:
            self._show_notice("Bấm Dự đoán để cập nhật kết quả và biểu đồ.")

    def _change_view(self, _change=None):
        titles = {"trend": "Xu hướng khối lượng", "hist": "Phần dư trên tập kiểm tra", "box": "Sai số theo loài"}
        self.figure_title.value = f"<h3>{titles[self.view.value]}</h3>"
        self.feature_picker.layout.display = "" if self.view.value == "trend" else "none"
        if self.rendered or self._rendering:
            self._render()

    def _render(self, _button=None):
        if not self.models or any(not self.controls[k].valid for k in self.numeric_features):
            return
        self._revision += 1
        revision = self._revision
        self._rendering = True
        self.predict_button.disabled = True
        self.predict_button.description = "Đang tính…"
        self.result.announcement = "Đang tính dự đoán và vẽ biểu đồ."
        try:
            name = self.model_picker.value
            inputs = {key: control.value for key, control in self.controls.items()}
            point = pd.DataFrame([inputs])
            if name not in self.models or any(inputs[key] is None for key in FEATURES):
                raise ValueError("Missing model or input.")
            if any(not self.controls[k].min <= inputs[k] <= self.controls[k].max for k in self.numeric_features):
                raise ValueError("Measurement outside the input range.")
            conformal = self.protocol.value == "conformal"
            support = self.proper if conformal else self.train
            predictors = ({n: fitted.predictor for n, fitted in self.calibrated.items()}
                          if conformal else self.models)
            names = [name] + [n for n in predictors if self.compare.value and n != name]
            estimate = {n: predictors[n].predict(point)[0] for n in names}
            if not np.isfinite(list(estimate.values())).all():
                raise ValueError("Non-finite prediction.")
            low_high = None
            info = self.performance.get(("full", name))
            radius = None
            if conformal:
                level = self.level.value
                radius = self.calibrated[name].radii[level]
                low_high = (estimate[name] - radius, estimate[name] + radius)
                info = self.performance[name, level]
            if self.view.value == "trend":
                feature = self.feature_picker.value
                control = self.controls[feature]
                values = ([v for _, v in control.options] if feature in CATEGORICAL
                          else np.linspace(control.min, control.max, 100))
                scenarios = point.loc[point.index.repeat(len(values))].assign(**{feature: values})
                trends = {n: predictors[n].predict(scenarios) for n in names}
                if not np.isfinite(list(trends.values())).all():
                    raise ValueError("Non-finite conditional prediction.")
                image, alt, caption = trend_plot(name, feature, self.labels, point, values, estimate,
                                                  trends, support, radius)
            else:
                image, alt, caption = error_plot(self.view.value, name,
                                                 self.residuals[self.protocol.value, name], self.test["species"])
            result = self._result_html(name, point, estimate, support, info, low_high)
            # Jupyter may deliver another widget event while Matplotlib draws.
            if revision != self._revision:
                return
            self.result.value = result
            self.result.announcement = f"{name}, khối lượng dự đoán {number(estimate[name])} g. {alt}"
            self.caption.value = f'<div lang="vi" class="pg-caption">{caption}</div>'
            with self.chart.hold_sync():
                self.chart.description = alt
                self.chart.value = image
            self.figure_panel.layout.display = ""
            self.rendered = True
        except ValueError:
            if revision != self._revision:
                return
            self._input_changed()
            self._show_notice("Không thể dự đoán. Hãy kiểm tra số đo, loài, giới tính và mô hình đã chọn.")
        finally:
            if revision == self._revision:
                self._rendering = False
                self.predict_button.description = "Dự đoán"
                self.predict_button.disabled = False

    def _result_html(self, name, point, estimate, support, info, bounds):
        safe_name = escape(name)
        prediction = number(estimate[name])
        band = ""
        detail = ""
        evidence = ""
        if bounds is not None:
            level = self.level.value
            low, high = map(number, bounds)
            band = (f'<div class="pg-band">Khoảng cá thể {level:.0%}<div class="pg-track" aria-hidden="true"></div>'
                    f'<span class="pg-bounds"><b>{low} g</b><span class="pg-sr-only">đến</span><b>{high} g</b></span></div>')
            evidence = (f'{info["n"] - info["outside"]}/{info["n"]} cá thể test nằm trong khoảng '
                        f'({number(100 * info["coverage"])}%). Sai số tuyệt đối trung bình {number(info["MAE (g)"])} g.')
            detail = (f'Mô hình học từ {len(support)} cá thể và hiệu chỉnh trên {len(self.train) - len(support)} cá thể khác. '
                      f'Mức {level:.0%} là mục tiêu bao phủ chung cho các cá thể có thể trao đổi được. '
                      f'Độ rộng {number(info["width_g"])} g cố định với mô hình và mức đã chọn. '
                      f'Tỷ lệ quan sát dùng {info["n"]} mẫu test, có thể khác giữa các loài.')
        else:
            if info is not None:
                evidence = f'Sai số tuyệt đối trung bình {number(info["MAE (g)"])} g trên {len(self.test)} cá thể test.'
            detail = f'Chế độ benchmark sử dụng toàn bộ {len(support)} mẫu train.'
            if name == "OLS":
                row = ols_prediction_intervals(self.models[name], self.ols_inference, point).iloc[0]
                band = (f'<div class="pg-band">PI 95% cho cá thể<div class="pg-track" aria-hidden="true"></div>'
                        f'<span class="pg-bounds"><b>{number(row.obs_ci_lower)} g</b><span class="pg-sr-only">đến</span>'
                        f'<b>{number(row.obs_ci_upper)} g</b></span></div>')
                evidence += (f' CI 95% cho trung bình từ {number(row.mean_ci_lower)} đến {number(row.mean_ci_upper)} g.')
                detail += ' PI cổ điển gồm thêm phương sai của cá thể mới và dựa trên giả định OLS.'
            elif self.calibrated:
                detail += ' Chọn chế độ dự đoán cá thể để xem khoảng conformal.'
        result = (f'<section class="pg-answer" lang="vi"><h3>{safe_name} · khối lượng dự đoán</h3>'
                  f'<div class="pg-values"><strong class="pg-mass">{prediction} g</strong>{band}</div>')
        if evidence:
            result += f'<div class="pg-evidence">{evidence}</div>'
        if len(estimate) > 1:
            other = " · ".join(f"{escape(n)} {number(v)} g" for n, v in estimate.items() if n != name)
            result += f'<p class="pg-alternatives">Cùng đầu vào · {other}</p>'
        result += f'<details><summary>Cách đọc kết quả</summary>{detail}</details></section>'
        species = point["species"].iloc[0]
        group = support.loc[support.species.eq(species)]
        outside = [escape(self.labels.get(k, k)) for k in self.numeric_features
                   if not group[k].min() <= point[k].iloc[0] <= group[k].max()]
        if outside:
            result += f'<p class="pg-warning">Ngoài khoảng train của loài {escape(species)} · {", ".join(outside)}.</p>'
        return result
