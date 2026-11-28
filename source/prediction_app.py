"""Notebook interface for exploring individual predictions and held-out errors."""
from base64 import b64encode
from html import escape
import csv
import json
from io import StringIO
from urllib.parse import quote
from pathlib import Path
from math import ceil, floor, isfinite

import anywidget
import ipywidgets as widgets
import numpy as np
import pandas as pd
import traitlets

from .analysis.uncertainty import LEVELS, MODULES, fit_calibrated, interval_summary, proper_calibration_split
from .evaluation import fit_ols_inference, ols_prediction_intervals
from .modeling import CATEGORICAL, FEATURES, TARGET, regression_metrics
from . import prediction_evidence as evidence
from .prediction_figures import error_plot, number, trend_plot

ASSETS = Path(__file__).with_name("ui")
MODEL_NAMES = {"OLS": "OLS · Tuyến tính", "M1": "M1 · Ridge", "M2": "M2 · Spline cơ sở",
               "M3": "M3 · Spline làm mượt", "M4": "M4 · Huber"}
SEX = {"female": "Cái", "male": "Đực"}


class Measurement(anywidget.AnyWidget):
    _esm = ASSETS / "measurement.js"
    _font = b64encode((ASSETS / "fonts/bricolage-600.woff2").read_bytes()).decode("ascii")
    _css = (f"@font-face{{font-family:PalmerDisplay;src:url(data:font/woff2;base64,{_font}) "
            "format('woff2');font-weight:600;font-display:swap}" + (ASSETS / "panel.css").read_text(encoding="utf-8"))
    description = traitlets.Unicode().tag(sync=True)
    field = traitlets.Unicode().tag(sync=True)
    min = traitlets.Float().tag(sync=True)
    max = traitlets.Float().tag(sync=True)
    step = traitlets.Float(.1).tag(sync=True)
    value = traitlets.Float().tag(sync=True)
    reset_count = traitlets.Int().tag(sync=True)
    valid = traitlets.Bool(True).tag(sync=True)


class Anatomy(anywidget.AnyWidget):
    _esm = ASSETS / "anatomy.js"
    svg = traitlets.Unicode((ASSETS / "anatomy.svg").read_text()).tag(sync=True)
    species = traitlets.Unicode().tag(sync=True)
    values = traitlets.Dict().tag(sync=True)


class FigurePicker(anywidget.AnyWidget):
    _esm = ASSETS / "figure-picker.js"
    fullscreen = traitlets.Bool().tag(sync=True)
    legend = traitlets.Unicode("Biểu đồ").tag(sync=True)
    options = traitlets.List().tag(sync=True)
    value = traitlets.Unicode("trend").tag(sync=True)


class Chart(anywidget.AnyWidget):
    _esm = ASSETS / "chart.js"
    value = traitlets.Bytes().tag(sync=True)
    compact = traitlets.Bytes().tag(sync=True)
    inspection = traitlets.Dict().tag(sync=True)
    description = traitlets.Unicode().tag(sync=True)


class Result(anywidget.AnyWidget):
    _esm = ASSETS / "result.js"
    value = traitlets.Unicode().tag(sync=True)
    announcement = traitlets.Unicode().tag(sync=True)
    reset_count = traitlets.Int().tag(sync=True)


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
        self.predictions = {}
        self.calibration_scores = {}
        self.benchmark = None
        self._figure_cache = {}
        self._recent_trends = {}
        self._batch = False
        if test is not None:
            if not test.index.intersection(train.index).empty:
                raise ValueError("Test rows overlap training rows.")
            self.proper, calibration = proper_calibration_split(train)
            self.calibrated = fit_calibrated(self.proper, calibration, [n for n in models if n in MODULES])
            for name, fitted in models.items():
                prediction = fitted.predict(test[FEATURES])
                self.predictions["full", name] = prediction
                self.residuals["full", name] = test[TARGET] - prediction
                self.performance["full", name] = regression_metrics(test[TARGET], prediction)
            for name, fitted in self.calibrated.items():
                self.calibration_scores[name] = np.abs(calibration[TARGET] - fitted.predictor.predict(calibration[FEATURES]))
                for level in LEVELS:
                    point, low, high = fitted.interval(test[FEATURES], level)
                    self.performance[name, level] = {
                        **regression_metrics(test[TARGET], point),
                        **interval_summary(test[TARGET], low, high),
                    }
                self.predictions["conformal", name] = point
                self.residuals["conformal", name] = test[TARGET] - point
            self.benchmark = evidence.verified_benchmark(test, {n: self.predictions["full", n] for n in models})

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
        views = [("Xu hướng", "trend")]
        if self.residuals:
            views.extend([("Phần dư", "hist"), ("Sai số theo loài", "box"), ("Thực tế và dự đoán", "actual")])
        self.view = FigurePicker(options=views)
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
                control = Measurement(field=feature, description=label, min=observed.min(), max=observed.max(),
                                      value=observed.median(), step=1 if feature == "year" else .1)
                control.observe(self._input_changed, names="valid")
            self.controls[feature] = control

        self.defaults = {key: c.value for key, c in self.controls.items()}
        self.presets = evidence.demo_rows(train)
        self.preset = widgets.Dropdown(description="Cá thể mẫu", options=["Tùy chỉnh", *self.presets]).add_class("pg-preset")
        self.reset_button = widgets.Button(description="Khôi phục mặc định").add_class("pg-reset")
        sections = [("Dự đoán cá thể", "predict")]
        if self.residuals:
            sections.append(("So sánh mô hình", "compare"))
        if self.calibrated:
            sections.extend([("Kiểm chứng khoảng", "verify"), ("Phương pháp", "method")])
        self.section = FigurePicker(fullscreen=True, options=sections, value="predict", legend="Không gian phân tích").add_class("pg-nav")
        self.reliability = FigurePicker(options=[("Cá thể test", "individual"), ("Bao phủ và độ rộng", "width")],
                                        value="individual", legend="Kiểm chứng khoảng")
        shore = b64encode((ASSETS / "palmer-shore.jpg").read_bytes()).decode("ascii")
        header = widgets.HTML(
            '<header class="pg-heading" lang="vi">'
            f'<img class="pg-shore" src="data:image/jpeg;base64,{shore}" alt="">'
            '<div><h2 lang="en">Palmer <em>Penguins</em></h2>'
            '<p>Ước lượng khối lượng từ ba số đo, loài và giới tính.</p></div></header>')
        self.anatomy = Anatomy()
        specimen = widgets.HTML('<h3>Cá thể khảo sát</h3>')
        self.form_predict = widgets.Button(description="Tính từ số đo").add_class("pg-form-predict")
        traitlets.dlink((self.predict_button, "disabled"), (self.form_predict, "disabled"))
        self.form_predict.on_click(self._render)
        jump = Result(value='<button type="button" class="pg-jump" data-target="predict">Đến nút dự đoán</button>'
                            '<button type="button" class="pg-jump" data-target="result">Đọc kết quả dự đoán</button>')
        self.form = widgets.VBox([
            specimen, self.preset,
            widgets.Box([self.controls["species"], self.controls["sex"]]).add_class("pg-pair"),
            *[self.controls[f] for f in self.numeric_features],
            self.form_predict, jump.add_class("pg-jump-host"),
        ]).add_class("pg-form")
        settings = widgets.Box([self.protocol, self.model_picker, self.level,
                                self.predict_button, self.reset_button]).add_class("pg-command")
        self.figure_title = widgets.HTML("<h3>Xu hướng khối lượng</h3>")
        self.figure_panel = widgets.VBox([
            self.figure_title, self.view, self.reliability, self.feature_picker, self.compare, self.chart, self.caption,
        ]).add_class("pg-view")
        self.body = widgets.Box([self.result.add_class("pg-summary"), self.form, self.anatomy, self.figure_panel]).add_class("pg-layout")
        self.glossary = Result(value=(ASSETS / "glossary.html").read_text(encoding="utf-8"))
        self.panel = widgets.VBox([header, self.section, settings, self.body, self.glossary],
                                  layout=widgets.Layout(width="100%", max_width="1140px", margin="0"))
        self.panel.add_class("penguin-panel")
        self.protocol.observe(self._change_protocol, names="value")
        self._change_protocol()
        for control in (self.level, self.model_picker, *self.controls.values()):
            control.observe(self._input_changed, names="value")
        for control in (self.view, self.compare, self.feature_picker):
            control.observe(self._change_view, names="value")
        self.section.observe(self._change_view, names="value")
        self.reliability.observe(self._change_view, names="value")
        self.preset.observe(self._apply_preset, names="value")
        self.reset_button.on_click(self._reset)
        self.predict_button.on_click(self._render)
        self._change_view()

    def _fill(self, values):
        for key, control in self.controls.items():
            control.value = values[key]
            if key in self.numeric_features:
                control.valid = True
                control.reset_count += 1

    def _apply_preset(self, change):
        if self._batch or change.new not in self.presets:
            return
        self._batch = True
        try:
            self._fill(self.train.loc[self.presets[change.new], FEATURES])
        finally:
            self._batch = False
        self._input_changed()

    def _reset(self, _button=None):
        self._batch = True
        try:
            self.protocol.value = self.protocol.options[0][1]
            self.model_picker.index = 0 if self.model_picker.options else None
            self.level.value = .95
            self.section.value = "predict"
            self.view.value = "trend"
            self.reliability.value = "individual"
            self.feature_picker.index = 0
            self.compare.value = len(self.models) > 1 and not self.calibrated
            self.preset.index = 0
            self._fill(self.defaults)
        finally:
            self._batch = False
        self._input_changed()
        self._change_view()
        self._render()
        for output in (self.result, self.caption, self.glossary):
            output.reset_count += 1

    def _change_protocol(self, _change=None):
        conformal = self.protocol.value == "conformal"
        names = self.calibrated if conformal else self.models
        current = self.model_picker.value
        self.model_picker.options = [(MODEL_NAMES.get(n, n), n) for n in names]
        self.model_picker.value = current if current in names else next(iter(names), None)
        self.level.disabled = not conformal
        self.compare.disabled = len(names) < 2
        self._input_changed()
        self._change_view()

    def _show_notice(self, text):
        self.result.value = (f'<section class="pg-notice" lang="vi"><h3>Khối lượng dự đoán</h3>'
                             f'<p>{escape(text)}</p></section>')
        self.result.announcement = text

    def _input_changed(self, _change=None):
        if self._batch:
            return
        if _change and _change.owner in self.controls.values() and self.preset.index:
            self.preset.index = 0
        self.anatomy.species = self.controls["species"].value or ""
        self.anatomy.values = {key: (self.controls[key].value if self.controls[key].valid else None)
                               for key in self.numeric_features}
        self._revision += 1
        self._rendering = False
        self.predict_button.description = "Dự đoán"
        self.rendered = False
        with self.chart.hold_sync():
            self.chart.value = b""
            self.chart.compact = b""
            self.chart.inspection = {}
        self.caption.value = '<p class="pg-plot-placeholder" lang="vi">Biểu đồ chờ kết quả dự đoán.</p>'
        invalid = any(not self.controls[key].valid for key in self.numeric_features)
        self.predict_button.disabled = invalid or not self.models
        if not self.models:
            self._show_notice("Chưa có mô hình sẵn sàng. Hãy chạy cell huấn luyện.")
        elif invalid:
            self._show_notice("Kiểm tra số đo được đánh dấu trước khi dự đoán.")
        else:
            self._show_notice("Bấm Dự đoán để cập nhật kết quả và biểu đồ.")

    def _change_view(self, _change=None):
        if self._batch:
            return
        section = self.section.value
        predict = section == "predict"
        trend = predict and self.view.value == "trend"
        conformal = self.protocol.value == "conformal"
        titles = {"trend": "Xu hướng khối lượng", "hist": "Phân bố phần dư", "box": "Sai số theo loài",
                  "actual": "Thực tế và dự đoán"}
        title = {"predict": titles[self.view.value],
                 "compare": "Khoảng dự đoán theo mô hình" if conformal else "Chênh lệch MAE so với OLS",
                 "verify": "Bao phủ và độ rộng" if self.reliability.value == "width" else "Khoảng dự đoán trên tập test",
                 "method": "Từ sai số hiệu chỉnh đến khoảng cá thể"}[section]
        self.figure_title.value = f'<h3 tabindex="-1">{title}</h3>'
        self.form.layout.display = "" if predict else "none"
        self.anatomy.layout.display = "" if predict else "none"
        if predict:
            self.body.remove_class("pg-analysis")
        else:
            self.body.add_class("pg-analysis")
        self.view.layout.display = "" if predict else "none"
        self.reliability.layout.display = "" if section == "verify" and conformal else "none"
        self.feature_picker.layout.display = "" if trend else "none"
        self.compare.layout.display = "" if trend else "none"
        if self.rendered or self._rendering:
            self._render()

    def _analysis_plot(self, point, name):
        section = self.section.value
        mode = self.protocol.value
        level = self.level.value
        if section == "compare" and mode == "conformal":
            rows = evidence.comparison_rows(self.calibrated, self.performance, point, level)
            return evidence.interval_plot(rows, level, name)
        # Individual intervals depend on the specimen. Population charts have a finite set of inputs.
        if section == "predict":
            key = (section, mode, name, self.view.value)
        elif section == "compare":
            key = (section, mode)
        else:
            key = (section, mode, name, level, self.reliability.value)
        if key not in self._figure_cache:
            if section == "compare":
                if self.benchmark is None:
                    return (b"", b"", {}), "Benchmark", "Bảng đối chiếu CV và bootstrap chưa có cho tập dữ liệu này."
                plot = evidence.benchmark_plot(self.benchmark, self.models, len(self.test))
            elif section in ("verify", "method") and mode != "conformal":
                return (b"", b"", {}), "Hướng dẫn", "Chọn chế độ Dự đoán cá thể · conformal và bấm Dự đoán để xem kết quả hiệu chỉnh."
            elif section == "method":
                plot = evidence.method_plot(self.calibration_scores[name], name, level, len(self.train), len(self.test))
            elif section == "verify" and self.reliability.value == "width":
                plot = evidence.coverage_plot(self.calibrated, self.performance, name)
            elif section == "verify":
                radius = self.calibrated[name].radii[level]
                rows = evidence.test_rows(self.test, self.predictions[mode, name], radius)
                plot = evidence.heldout_plot(rows, name, level)
            elif self.view.value == "actual":
                rows = evidence.test_rows(self.test, self.predictions[mode, name])
                info = self.performance[name, level] if mode == "conformal" else self.performance["full", name]
                plot = evidence.actual_plot(rows, name, info, mode if mode == "conformal" else "benchmark")
            else:
                plot = error_plot(self.view.value, name, self.residuals[mode, name], self.test["species"])
            self._figure_cache[key] = plot
        return self._figure_cache[key]

    def _render(self, _button=None):
        if self._batch or not self.models or any(not self.controls[k].valid for k in self.numeric_features):
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
            names = [name] + [n for n in predictors if self.compare.value and self.section.value == "predict" and self.view.value == "trend" and n != name]
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
            if self.section.value == "predict" and self.view.value == "trend":
                feature = self.feature_picker.value
                control = self.controls[feature]
                values = ([v for _, v in control.options] if feature in CATEGORICAL
                          else np.linspace(control.min, control.max, 100))
                key = (self.protocol.value, tuple(names), self.level.value, feature, tuple(inputs.values()))
                if key not in self._recent_trends:
                    scenarios = point.loc[point.index.repeat(len(values))].assign(**{feature: values})
                    trends = {n: predictors[n].predict(scenarios) for n in names}
                    if not np.isfinite(list(trends.values())).all():
                        raise ValueError("Non-finite conditional prediction.")
                    drawn = trend_plot(name, feature, self.labels, point, values, estimate, trends, support, radius)
                    if revision != self._revision:
                        return
                    if len(self._recent_trends) >= 6:
                        del self._recent_trends[next(iter(self._recent_trends))]
                    self._recent_trends[key] = drawn
                image, alt, caption = self._recent_trends[key]
            else:
                image, alt, caption = self._analysis_plot(point, name)
            result, interval_text = self._result_html(name, point, estimate, support, info, low_high)
            # Jupyter may deliver another widget event while Matplotlib draws.
            if revision != self._revision:
                return
            self.result.value = result
            spoken = f"{name}, khối lượng dự đoán {number(estimate[name])} g."
            self.result.announcement = f"{spoken} {interval_text} {alt}".strip()
            help_key = self.section.value if self.section.value != "predict" else self.view.value
            help_text = {
                "compare": ("Độ rộng và MAE", "Độ rộng là khoảng cách giữa hai cận. MAE là trung bình độ lệch tuyệt đối trên test.") if conformal else ("ΔMAE và CI bootstrap", "Mỗi lượt bootstrap lấy lại cùng ID cho cả hai mô hình. CI chứa 0 nếu dữ liệu tương thích với chênh lệch bằng 0."),
                "verify": ("Mục tiêu và bao phủ thực nghiệm", "Mục tiêu là mức chọn trước khi hiệu chỉnh. Bao phủ thực nghiệm là tỷ lệ khối lượng thật nằm giữa hai cận trên các cá thể test."),
                "method": ("Hiệu chỉnh khoảng", "Xếp sai số tuyệt đối trên các mẫu hiệu chỉnh độc lập với bước fit. Hạng được chọn xác định bán kính."),
                "actual": ("MAE, RMSE và R²", "MAE là độ lớn sai lệch trung bình. RMSE cho trọng số lớn hơn với sai lệch lớn. R² so sai số bình phương với mốc dự đoán trung bình."),
                "hist": ("Dấu của phần dư", "Thực tế trừ dự đoán. Phần dư dương là trường hợp mô hình dự đoán thấp."),
                "box": ("Cách đọc hộp", "Hai mép hộp là Q1 và Q3. Vạch giữa là trung vị sai số tuyệt đối của loài."),
                "trend": ("Cách đọc xu hướng", "Mỗi dự đoán trên đường chỉ thay một biến. Điểm quan sát train có thể khác nhau ở cả năm biến."),
            }[help_key]
            self.caption.value = (f'<div lang="vi" class="pg-caption">{caption}'
                                  f'<details class="pg-help"><summary>{help_text[0]}</summary>{help_text[1]}</details></div>')
            with self.chart.hold_sync():
                self.chart.description = alt
                self.chart.value, self.chart.compact, self.chart.inspection = image
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

    def _interval_html(self, title, bounds, point):
        low, high = bounds
        numbers = f'<span class="pg-bounds">{number(low)} g đến {number(high)} g</span>'
        if not all(isfinite(v) for v in bounds):
            return f'<div class="pg-band">{title}{numbers}</div>'
        # All models use the same graduated mass axis. Extend it for extrapolated limits.
        axis_min = 500 * floor(min(self.train[TARGET].min(), low) / 500)
        axis_max = max(axis_min + 500, 500 * ceil(max(self.train[TARGET].max(), high) / 500))
        span = axis_max - axis_min
        left = 100 * (low - axis_min) / span
        width = 100 * (high - low) / span
        mark = 100 * (point - axis_min) / span
        return (f'<div class="pg-band"><div>{title}</div>{numbers}'
                f'<div class="pg-track" aria-hidden="true"><i style="left:{left:.4f}%;width:{width:.4f}%"></i>'
                f'<b style="left:{mark:.4f}%"></b></div><div class="pg-scale">'
                f'<span>{number(axis_min, 0)} g</span><span>{number(axis_max, 0)} g</span></div></div>')

    def _result_html(self, name, point, estimate, support, info, bounds):
        safe_name = escape(name)
        prediction = number(estimate[name])
        band = ""
        interval_text = ""
        detail = ""
        ci = None
        evidence = ""
        if bounds is not None:
            level = self.level.value
            interval_text = f"Khoảng cá thể {level:.0%} từ {number(bounds[0])} đến {number(bounds[1])} g."
            band = self._interval_html(f"Khoảng cá thể {level:.0%}", bounds, estimate[name])
            evidence = (f'<div><dt>Bao phủ thực nghiệm</dt><dd>{number(100 * info["coverage"])}%'
                        f'<small>{info["n"] - info["outside"]}/{info["n"]} cá thể test nằm trong khoảng</small></dd></div>')
            detail = (f'Mô hình học từ {len(support)} cá thể và hiệu chỉnh trên {len(self.train) - len(support)} cá thể khác. '
                      f'Mức {level:.0%} là mục tiêu bao phủ chung cho các cá thể có thể trao đổi được. '
                      f'Độ rộng {number(info["width_g"])} g cố định với mô hình và mức đã chọn. '
                      f'Tỷ lệ quan sát dùng {info["n"]} mẫu test, có thể khác giữa các loài.')
        else:
            detail = f'Chế độ benchmark sử dụng toàn bộ {len(support)} mẫu train.'
            if name == "OLS":
                row = ols_prediction_intervals(self.models[name], self.ols_inference, point).iloc[0]
                bounds = (row.obs_ci_lower, row.obs_ci_upper)
                ci = (row.mean_ci_lower, row.mean_ci_upper)
                interval_text = f"PI 95% cho cá thể từ {number(row.obs_ci_lower)} đến {number(row.obs_ci_upper)} g."
                band = self._interval_html("PI 95% cho cá thể", (row.obs_ci_lower, row.obs_ci_upper), estimate[name])
                evidence = (f'<div><dt>CI 95% cho trung bình</dt><dd>{number(row.mean_ci_lower)} '
                            f'đến {number(row.mean_ci_upper)} g</dd></div>')
                detail += ' PI cổ điển gồm thêm phương sai của cá thể mới và dựa trên giả định OLS.'
            elif self.calibrated:
                detail += ' Chọn chế độ dự đoán cá thể để xem khoảng conformal.'
        if info is not None:
            evidence += (f'<div><dt>Sai số tuyệt đối trung bình</dt><dd>{number(info["MAE (g)"])} g'
                         f'<small>Trên {len(self.test)} cá thể test</small></dd></div>')
        species = point["species"].iloc[0]
        sex = point["sex"].iloc[0]
        result = (f'<section class="pg-answer" lang="vi"><div class="pg-readout"><div class="pg-point">'
                  f'<h3 tabindex="-1">{safe_name} · khối lượng dự đoán</h3>'
                  f'<strong class="pg-mass">{prediction} <small>g</small></strong>'
                  f'<span>{escape(species)} · {escape(SEX.get(sex, sex))}</span></div>{band}'
                  f'<dl class="pg-evidence">{evidence}</dl></div>')
        if len(estimate) > 1:
            other = " · ".join(f"{escape(n)} {number(v)} g" for n, v in estimate.items() if n != name)
            result += f'<p class="pg-alternatives">Cùng đầu vào · {other}</p>'
        group = support.loc[support.species.eq(species)]
        outside = [escape(self.labels.get(k, k)) for k in self.numeric_features
                   if not group[k].min() <= point[k].iloc[0] <= group[k].max()]
        warning = f"Ngoài khoảng train của loài {species} · {', '.join(outside)}." if outside else ""
        conformal = self.protocol.value == "conformal"
        record = {**point.iloc[0].to_dict(), "model": name, "protocol": self.protocol.value,
                  "train_n": len(support), "prediction_g": estimate[name],
                  "interval_type": "split conformal" if conformal else "OLS classical PI" if bounds else "",
                  "nominal_coverage": self.level.value if conformal else .95 if bounds else "",
                  "lower_g": bounds[0] if bounds else "", "upper_g": bounds[1] if bounds else "",
                  "width_g": bounds[1]-bounds[0] if bounds else "",
                  "mean_ci_lower_g": ci[0] if ci else "", "mean_ci_upper_g": ci[1] if ci else "",
                  "empirical_coverage": (info or {}).get("coverage", ""), "test_n": len(self.test) if info else "",
                  **{metric: (info or {}).get(metric, "") for metric in ("MAE (g)", "RMSE (g)", "R²")},
                  "interpretation": detail, "warning": warning}
        stream = StringIO()
        writer = csv.DictWriter(stream, fieldnames=record)
        writer.writeheader()
        writer.writerow(record)
        measurements = {key: float(point[key].iloc[0]) for key in self.numeric_features}
        export = (f'<a class="pg-export" data-measurements="{escape(json.dumps(measurements))}" '
                  f'download="palmer-{safe_name}.csv" '
                  f'href="data:text/csv;charset=utf-8,{quote(chr(0xfeff)+stream.getvalue())}">Tải CSV</a>')
        detail = f"Khối lượng dự đoán {prediction} g. " + detail
        detail += " MAE là độ lớn sai số trung bình theo gram trên tập test."
        if conformal:
            detail += " Mức mục tiêu và tỷ lệ bao phủ quan sát đều mô tả nhiều cá thể. Với cá thể đang nhập, hãy đọc hai cận theo gram."
        if self.section.value != "predict":
            mode = f"Conformal {self.level.value:.0%}" if conformal else "Benchmark"
            result = (f'<section class="pg-answer pg-context" lang="vi"><div><h3 tabindex="-1">{safe_name} · {mode}</h3>'
                      f'<span>{escape(species)} · {escape(SEX.get(sex, sex))} · </span>'
                      f'<strong class="pg-mass">{prediction} <small>g</small></strong></div>'
                      '<button type="button" class="pg-edit">Chỉnh số đo</button>')
            if ci:
                detail += f" CI 95% cho trung bình {number(ci[0])} đến {number(ci[1])} g. {interval_text}"
        result += (f'<div class="pg-reading"><details><summary>Cách đọc kết quả</summary>{detail}</details>{export}'
                   '<button type="button" class="pg-jump" data-target="chart">Xem biểu đồ</button></div></section>')
        if warning:
            result += f'<p class="pg-warning">{warning}</p>'
        return result, interval_text
