"""Optional browser acceptance check for section 5 in a running local JupyterLab."""
import argparse
import csv
import time
from urllib.parse import urlparse
from pathlib import Path
import re

from playwright.sync_api import expect, sync_playwright


def exercise(page, url, axe_path, evidence_dir=None, presentation=False, touch=False):
    page_errors = []
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    external = []
    if presentation:
        def offline(route):
            if urlparse(route.request.url).netloc not in ("", urlparse(url).netloc) and not route.request.url.startswith("blob:"):
                external.append(route.request.url)
                route.abort()
            else:
                route.continue_()
        page.route("**/*", offline)
    start = time.monotonic()
    page.goto(url, wait_until="domcontentloaded")
    if not presentation:
        page.wait_for_function("""() => {
            const w = window.jupyterapp?.shell.currentWidget;
            return w?.context?.path?.endsWith('.ipynb') &&
                   w.sessionContext?.session?.kernel?.connectionStatus === 'connected';
        }""", timeout=120_000)
        page.evaluate("""async () => {
            const app = window.jupyterapp, w = app.shell.currentWidget;
            await w.sessionContext.ready;
            await w.sessionContext.session.kernel.info;
            await app.commands.execute('apputils:change-theme', {theme:'JupyterLab Light'});
            app.shell.collapseLeft();
            await app.commands.execute('notebook:run-all-cells');
            const i = w.content.widgets.findIndex(cell => cell.model.sharedModel.getSource().includes('display(prediction_app.panel)'));
            if (i < 0) throw new Error('Missing prediction application cell');
            await w.content.scrollToItem(i);
        }""")
    app = page.get_by_role("region", name="Dự đoán khối lượng chim cánh cụt")
    expect(app.locator(".pg-mass")).to_have_text("3.571,7 g", timeout=120_000)
    expect(app.locator(".pg-bounds")).to_have_text("2.935,5 g đến 4.208,0 g")
    print("Open to first prediction (s)", round(time.monotonic()-start,2))
    page.add_script_tag(path=str(axe_path))

    def audit(state):
        findings = app.evaluate("""async node => {
            const tags = ['wcag2a','wcag2aa','wcag21a','wcag21aa','wcag22aa'];
            const r = await axe.run(node, {runOnly:{type:'tag',values:tags}});
            return {violations:r.violations.map(x=>[x.id,x.nodes.map(n=>n.target)]),
                    incomplete:r.incomplete.map(x=>[x.id,x.nodes.map(n=>n.target)])};
        }""")
        print(state, findings)
        assert not findings["violations"], findings

    button = app.get_by_role("button", name="Dự đoán", exact=True)
    result = app.locator(".pg-answer")
    plot = app.locator(".pg-chart img")
    caption = app.locator(".pg-caption")
    def pick(title):
        return app.get_by_label(title, exact=True)

    nav = app.get_by_role("group", name="Không gian phân tích")
    field = app.get_by_role("spinbutton").first
    expect.set_options(timeout=12000)

    def shot(key, element=None):
        if evidence_dir is None:
            return
        evidence_dir.mkdir(parents=True, exist_ok=True)
        size = page.viewport_size
        height = min(4200, round(app.bounding_box()["height"]) + 360)
        page.set_viewport_size({"width": size["width"], "height": height})
        app.scroll_into_view_if_needed()
        target = app if element is None else element
        target.screenshot(path=evidence_dir / (key + ".png"))
        page.set_viewport_size(size)

    def all_sizes():
        current = page.viewport_size
        for w,h in ((1920,1080),(1440,900),(1366,768),(1280,720),(768,900),(430,844),(390,844),(320,720)):
            viewport_check(w,h)
        page.set_viewport_size(current)

    def show(group, title, expected):
        group.locator("label").filter(has_text=re.compile("^" + re.escape(title) + "$")).click()
        expect(plot).to_have_attribute("alt", re.compile(expected))
        all_sizes()

    def predict():
        button.click()
        expect(result).to_be_visible()
        expect(button).to_be_enabled()

    def select(label, index):
        pick(label).select_option(index=index)
        expect(result).to_have_count(0)
        expect(plot).not_to_be_visible()
        predict()

    def rows():
        details = caption.locator("details").first
        if not details.evaluate("node => node.open"):
            details.locator("summary").click()
        return details.locator("tbody tr")

    def viewport_check(width, height):
        page.set_viewport_size({"width": width, "height": height})
        page.wait_for_timeout(150)
        assert app.evaluate("node => node.scrollWidth <= node.clientWidth + 1"), width

    audit("desktop")
    shot("prediction-dashboard")
    for width,height in ((1920,1080),(1440,900),(1366,768),(1280,720)):
        viewport_check(width,height)
        app.evaluate("node => node.scrollIntoView({block:'start'})")
        page.wait_for_timeout(200)
        print("viewport",width,height,"button",button.bounding_box(),"result",app.locator(".pg-mass").bounding_box())
        assert button.bounding_box()["y"] >= 0 and button.bounding_box()["y"]+button.bounding_box()["height"] < height
        assert app.locator(".pg-mass").bounding_box()["y"] < height
        if evidence_dir:
            page.screenshot(path=evidence_dir / f"viewport-{width}.png")
    viewport_check(1440,900)
    atlas = app.locator(".pg-atlas")
    expect(atlas).to_have_attribute("data-species","Adelie")
    for i in (1,2,0):
        pick("Loài").select_option(index=i)
        expect(atlas).to_have_attribute("data-species",pick("Loài").input_value())
    for key in ("bill_length_mm","bill_depth_mm","flipper_length_mm"):
        selected_field=app.locator(f'[data-field="{key}"] input[type=number]')
        selected_field.focus()
        mark=atlas.locator(f'[data-measure="{key}"]')
        expect(mark).to_have_class(re.compile("pg-chosen"))
        mark.focus()
        mark.press("Enter")
        expect(selected_field).to_be_focused()
        if touch:
            mark.tap()
            expect(selected_field).to_be_focused()
    pick("Loài").select_option(index=2)
    expect(atlas).to_have_attribute("data-species","Gentoo")
    app.locator('[data-field="bill_depth_mm"] label').hover()
    expect(atlas.locator(".pg-chosen")).to_have_attribute("data-measure","bill_depth_mm")
    shot("anatomy",atlas)
    app.get_by_role("button",name="Khôi phục mặc định").click()
    expect(result).to_contain_text("3.571,7 g")
    export=app.get_by_role("link",name="Tải CSV")
    with page.expect_download() as saving:
        export.click()
    with open(saving.value.path(), encoding="utf-8-sig", newline="") as stream:
        row=next(csv.DictReader(stream))
    assert row["species"]=="Adelie" and row["protocol"]=="conformal" and row["model"]=="OLS"
    assert abs(float(row["prediction_g"])-3571.7)<.1 and abs(float(row["lower_g"])-2935.5)<.1
    assert abs(float(row["empirical_coverage"])-67/69)<1e-9
    field.fill("197")
    expect(export).to_be_visible()
    field.fill("196")
    expect(export).to_have_count(0)
    app.get_by_role("button",name="Khôi phục mặc định").click()
    expect(result).to_contain_text("3.571,7 g")
    full=app.get_by_role("button",name="Toàn màn hình",exact=True)
    if full.count():
        full.click()
        page.wait_for_function("document.fullscreenElement?.classList.contains('penguin-panel')")
        app.get_by_role("button", name="Thoát toàn màn hình").click()
        page.wait_for_function("!document.fullscreenElement")
        expect(full).to_be_focused()
    if presentation:
        assert page.locator(".jp-CodeCell .jp-InputArea-editor:visible").count()==0
        page.screenshot(path=evidence_dir/"presentation.png") if evidence_dir else None
    preset = pick("Cá thể mẫu")
    for i in range(1,preset.locator("option").count()):
        select("Cá thể mẫu",i)
        expect(app.locator(".pg-point")).to_contain_text(pick("Loài").input_value())
    app.get_by_role("button",name="Khôi phục mặc định").click()
    expect(app.locator(".pg-mass")).to_have_text("3.571,7 g")
    for i in range(1,4):
        select("Mô hình",i)
        expect(result).to_contain_text(f"M{i} ·")
    for slider in app.get_by_role("slider").all():
        slider.focus()
        slider.press("ArrowRight")
        expect(result).to_have_count(0)
        predict()
    field.fill("999")
    expect(button).to_be_disabled()
    expect(field).to_have_attribute("aria-invalid","true")
    audit("invalid")
    shot("input-error")
    app.get_by_role("button",name="Khôi phục mặc định").click()
    expect(field).to_have_value("197")
    expect(button).to_be_enabled()
    expect(app.locator(".pg-mass")).to_have_text("3.571,7 g")
    field.fill("190")
    expect(result).to_have_count(0)
    predict()
    select("Loài",2)
    expect(app.locator(".pg-warning")).to_contain_text("Gentoo")
    select("Giới tính",1)
    app.get_by_role("checkbox").check()
    expect(result).to_contain_text("Cùng đầu vào")
    pick("Biến trên biểu đồ").select_option(index=4)
    expect(plot).to_have_attribute("alt",re.compile("Giới tính"))
    app.get_by_role("checkbox").uncheck()
    expect(result).not_to_contain_text("Cùng đầu vào")
    app.get_by_role("button",name="Khôi phục mặc định").click()
    expect(result).to_contain_text("3.571,7 g")

    show(nav,"So sánh mô hình","Conformal 95%")
    expect(rows()).to_have_count(4)
    expect(rows().first).to_contain_text("2.935,5")
    expect(rows().first).to_contain_text("1.272,6")
    audit("intervals")
    shot("interval-comparison")
    select("Mức bao phủ",0)
    expect(rows().first).to_contain_text("1.164,9")
    select("Mô hình",2)
    show(nav,"Kiểm chứng khoảng","M2 · mục tiêu 90%")
    inspector=app.get_by_label("Điểm dữ liệu",exact=True)
    expect(inspector.locator("option")).to_have_count(69)
    checked = rows()
    expect(checked).to_have_count(69)
    assert checked.locator("td:last-child").all_inner_texts().count("Ngoài")==7
    expect(caption).to_contain_text("62/69")
    idx=checked.locator("td:last-child").all_inner_texts().index("Ngoài")
    expected_id=checked.nth(idx).locator("td").first.inner_text()
    inspector.select_option(index=idx)
    expect(app.locator(".pg-tip")).to_contain_text("Ngoài khoảng")
    expect(inspector.locator("option:checked")).to_contain_text("ID "+expected_id)
    stage = app.locator(".pg-plot-stage")
    coords = app.locator(".pg-halo").evaluate("""e => ({
        x: e.parentElement.clientWidth * parseFloat(e.style.left) / 100,
        y: e.parentElement.clientHeight * parseFloat(e.style.top) / 100
    })""")
    inspector.select_option(index=(idx+1)%69)
    stage.hover(position=coords)
    expect(inspector.locator("option:checked")).to_contain_text("ID "+expected_id)
    expect(stage.locator(".pg-point-tag")).to_contain_text("ID "+expected_id)
    if touch:
        inspector.select_option(index=(idx+1)%69)
        stage.tap(position=coords)
        expect(inspector.locator("option:checked")).to_contain_text("ID "+expected_id)
    audit("test-intervals")
    caption.locator("summary").first.click()
    shot("test-intervals")
    shot("test-plot", app.locator(".pg-view"))
    stage.hover(position=coords)
    shot("point-inspector", app.locator(".pg-chart"))
    expand=app.get_by_role("button",name="Mở rộng biểu đồ")
    original_width = stage.bounding_box()["width"]
    expand.click()
    dialog=page.get_by_role("dialog",name="Biểu đồ mở rộng")
    expect(dialog).to_be_visible()
    expect(dialog.get_by_role("heading")).to_have_text("Khoảng dự đoán trên tập test")
    expect(dialog.get_by_role("button",name="Đóng biểu đồ")).to_be_focused()
    assert dialog.locator(".pg-plot-stage").bounding_box()["width"] > original_width
    page.keyboard.press("Tab")
    assert dialog.evaluate("d=>d.contains(document.activeElement)")
    expect(dialog.locator(".pg-tip")).to_contain_text("Ngoài khoảng")
    audit("expanded")
    if evidence_dir:
        dialog.screenshot(path=evidence_dir/"expanded.png")
    page.keyboard.press("Escape")
    expect(dialog).not_to_be_visible()
    expect(expand).to_be_focused()
    expect(inspector.locator("option:checked")).to_contain_text("ID "+expected_id)
    help_box = caption.locator(".pg-help")
    help_box.locator("summary").press("Enter")
    expect(help_box).to_contain_text("tỷ lệ khối lượng thật")
    help_box.locator("summary").press("Enter")
    checks=app.get_by_role("group",name="Kiểm chứng khoảng",exact=True)
    show(checks,"Bao phủ và độ rộng","M2: từ 90% lên 95%")
    expect(rows()).to_have_count(8)
    expect(caption).to_contain_text("7 và 1")
    audit("coverage-width")
    shot("coverage-width")
    show(nav,"Phương pháp","hạng 51/55")
    select("Mức bao phủ",1)
    expect(caption).to_contain_text("hạng 54/55")
    expect(rows()).to_have_count(55)
    caption.locator("summary").first.click()
    glossary=app.get_by_text("Tra cứu thuật ngữ thống kê",exact=True)
    glossary.focus()
    glossary.press("Enter")
    expect(app.locator(".pg-glossary dt")).to_have_count(18)
    audit("method")
    glossary.press("Enter")
    shot("method")
    show(nav,"So sánh mô hình","Conformal 95%")
    select("Chế độ",1)
    expect(plot).to_have_attribute("alt",re.compile("3/3 khoảng bootstrap"))
    expect(rows()).to_have_count(4)
    expect(rows().first).to_contain_text("256,7")
    audit("benchmark")
    shot("benchmark")
    for i in (0,1,2,3):
        select("Mô hình",i)
        if i==0:
            expect(result).to_contain_text("CI 95%")
            expect(app.locator("[role=status]").first).to_contain_text("PI 95%")
    show(nav,"Dự đoán cá thể","Dự đoán theo")
    figures=app.get_by_role("group",name="Biểu đồ",exact=True)
    show(figures,"Thực tế và dự đoán","M3 · benchmark")
    expect(rows()).to_have_count(69)
    expect(caption).to_contain_text("RMSE 307,0")
    audit("actual")
    caption.locator("summary").first.click()
    shot("actual-predicted")
    select("Chế độ",0)
    expect(plot).to_have_attribute("alt",re.compile("M3 · conformal"))
    expect(caption).to_contain_text("258,2 g")
    show(figures,"Sai số theo loài","Sai số tuyệt đối")
    assert sum(map(int,rows().locator("td:nth-child(2)").all_inner_texts()))==69
    audit("species-errors")
    shot("prediction-errors")
    show(figures,"Phần dư","Histogram")
    assert sum(map(int,rows().locator("td:last-child").all_inner_texts()))==69
    shot("residual-histogram")
    app.get_by_role("button",name="Khôi phục mặc định").click()
    expect(app.locator(".pg-mass")).to_have_text("3.571,7 g")
    page.emulate_media(reduced_motion="reduce")
    assert app.locator(".pg-answer").evaluate("e=>getComputedStyle(e).animationName")=="none"
    for width,height in ((768,900),(390,844),(320,768)):
        viewport_check(width,height)
        assert pick("Mô hình").evaluate("""select => {
            const css = getComputedStyle(select), context = document.createElement('canvas').getContext('2d');
            context.font = css.font;
            const longest = Math.max(...Array.from(select.options, o => context.measureText(o.text).width));
            return longest <= select.clientWidth - parseFloat(css.paddingLeft) - parseFloat(css.paddingRight) - 14;
        }"""), width
        reset = app.get_by_role("button", name="Khôi phục mặc định")
        assert reset.evaluate("b=>b.scrollWidth<=b.clientWidth+1 && b.scrollHeight<=b.clientHeight+1"), width
    viewport_check(390,844)
    if touch:
        field.focus()
        atlas.locator('[data-measure="bill_length_mm"] text').tap()
        expect(app.locator('[data-field="bill_length_mm"] input[type=number]')).to_be_focused()
        chip = stage.locator(".pg-halo")
        target = chip.evaluate("""e=>({x:e.parentElement.clientWidth*parseFloat(e.style.left)/100,
                                       y:e.parentElement.clientHeight*parseFloat(e.style.top)/100})""")
        inspector.select_option(index=2)
        stage.tap(position=target)
        expect(inspector.locator("option:checked")).to_contain_text("cá thể đang khảo sát")
    field.fill("196")
    expect(result).to_have_count(0)
    go_to_predict = app.get_by_role("button", name="Đến nút dự đoán")
    go_to_predict.focus()
    go_to_predict.press("Enter")
    expect(button).to_be_focused()
    button.press("Enter")
    expect(result).to_be_visible()
    app.get_by_role("button",name="Khôi phục mặc định").click()
    expect(app.locator(".pg-mass")).to_have_text("3.571,7 g")
    app.get_by_role("button",name="Xem biểu đồ").click()
    expect(figures.get_by_role("radio",name="Xu hướng")).to_be_focused()
    page.keyboard.press("ArrowRight")
    expect(plot).to_have_attribute("alt",re.compile("Histogram"))
    expect(plot).to_have_attribute("data-layout","compact")
    audit("mobile")
    show(figures,"Xu hướng","Dự đoán theo")
    shot("prediction-narrow")
    viewport_check(1366,768)
    app.evaluate("e=>e.style.zoom='200%'")
    assert app.evaluate("e=>e.scrollWidth<=e.clientWidth+1")
    expect(button).to_be_visible()
    app.evaluate("e=>e.style.removeProperty('zoom')")
    page.emulate_media(forced_colors="active")
    field.fill("")
    expect(button).to_be_disabled()
    field.fill("197")
    expect(button).to_be_enabled()
    button.focus()
    expect(button).to_be_focused()
    page.keyboard.press("Enter")
    expect(result).to_contain_text("3.571,7 g")
    page.emulate_media(forced_colors="none")
    predict()
    if presentation:
        page.emulate_media(color_scheme="dark")
    else:
        page.evaluate("""async () => {
            await window.jupyterapp.commands.execute('apputils:change-theme', {theme:'JupyterLab Dark'});
        }""")
    audit("dark")
    show(nav, "So sánh mô hình", "Conformal 95%")
    app.get_by_role("button", name="Chỉnh số đo").click()
    expect(preset).to_be_focused()
    expect(result).to_contain_text("3.571,7 g")
    assert not page_errors, page_errors
    assert not external, external
    print("All application views, four models, two protocols, presets, reset and keyboard flows passed.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="Local JupyterLab URL for a disposable copy of lab01.ipynb")
    parser.add_argument("--axe", type=Path, required=True, help="Local axe-core distribution (axe.min.js)")
    parser.add_argument("--presentation", action="store_true", help="Test the local Voilà launcher and block external HTTP")
    parser.add_argument("--touch", action="store_true", help="Exercise chart and anatomy through touch input")
    parser.add_argument("--browser", choices=("chromium", "firefox", "webkit"), default="chromium")
    parser.add_argument("--evidence-dir", type=Path, help="Optional folder for live screenshots")
    args = parser.parse_args()
    with sync_playwright() as playwright:
        browser = (playwright.chromium.launch(channel="chromium") if args.browser == "chromium"
                   else getattr(playwright, args.browser).launch())
        try:
            print(args.browser, browser.version)
            exercise(browser.new_page(viewport={"width": 1440, "height": 900}, has_touch=args.touch),
                     args.url, args.axe, args.evidence_dir, args.presentation, args.touch)
        finally:
            browser.close()
