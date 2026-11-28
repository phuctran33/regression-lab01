"""Time the live prediction panel in a disposable, freshly opened JupyterLab notebook."""
import argparse
import json
import re
from time import perf_counter

from playwright.sync_api import expect, sync_playwright


def measure(page, url, repetitions):
    opened = perf_counter()
    page.goto(url)
    page.wait_for_function("""() => {
        const w = window.jupyterapp?.shell?.currentWidget;
        return w?.context?.path?.endsWith('.ipynb') &&
               w.sessionContext?.session?.kernel?.connectionStatus === 'connected';
    }""", timeout=120_000)
    page.evaluate("""async () => {
        const app=window.jupyterapp, w=app.shell.currentWidget;
        await w.sessionContext.ready; await w.sessionContext.session.kernel.info; app.shell.collapseLeft();
    }""")
    run = perf_counter()
    page.evaluate("""async () => {
        const app=window.jupyterapp, w=app.shell.currentWidget;
        await app.commands.execute('notebook:run-all-cells');
        const i=w.content.widgets.findIndex(c=>c.model.sharedModel.getSource().includes('display(prediction_app.panel)'));
        await w.content.scrollToItem(i);
    }""")
    app = page.locator(".penguin-panel")
    mass = app.locator(".pg-mass")
    expect(mass).to_have_text("3.571,7 g", timeout=120_000)
    button = app.get_by_role("button", name="Dự đoán", exact=True)
    expect(button).to_be_enabled()
    ready = perf_counter()
    times = {"open_to_interactive_s": round(ready-opened, 3), "run_all_to_result_s": round(ready-run, 3)}
    result = app.locator(".pg-answer")
    plot = app.locator(".pg-chart img").first

    def predict():
        button.evaluate("b=>b.click()")
        result.wait_for()
        expect(button).to_be_enabled()

    def change_model():
        app.get_by_label("Mô hình", exact=True).select_option(index=2)
        result.wait_for(state="detached")
        predict()

    def tab(title, alt):
        app.locator(".pg-nav label").filter(has_text=title).evaluate("n=>n.click()")
        expect(plot).to_have_attribute("alt", re.compile(alt))

    def restore():
        app.get_by_role("button", name="Khôi phục mặc định").evaluate("b=>b.click()")
        expect(plot).to_have_attribute("alt", re.compile("Dự đoán theo"))
        expect(mass).to_have_text("3.571,7 g")

    for _ in range(repetitions):
        app.get_by_role("spinbutton").first.fill("196")
        result.wait_for(state="detached")
        for label, action in (("predict", predict), ("model_and_predict", change_model),
                              ("compare_tab", lambda: tab("So sánh mô hình", "Conformal")),
                              ("method_chart", lambda: tab("Phương pháp", "hạng")), ("reset", restore)):
            started = perf_counter()
            action()
            times.setdefault(label+"_ms", []).append(round(1000*(perf_counter()-started)))

    if app.locator(".pg-plot-stage").count():
        tab("Kiểm chứng khoảng", "67/69")
        stage = app.locator(".pg-plot-stage")
        stage.scroll_into_view_if_needed()
        picker = app.get_by_label("Điểm dữ liệu", exact=True)
        for index in range(1, min(repetitions + 1, 69)):
            picker.select_option(index=index)
            point = stage.locator(".pg-halo").evaluate("e=>[parseFloat(e.style.left)/100,parseFloat(e.style.top)/100]")
            picker.select_option(index=0)
            delay = stage.evaluate("""async (stage,[x,y])=>{
                const box=stage.getBoundingClientRect(), start=performance.now();
                stage.dispatchEvent(new PointerEvent('pointermove', {
                    clientX:box.left+x*box.width, clientY:box.top+y*box.height
                }));
                await new Promise(requestAnimationFrame);
                return performance.now()-start;
            }""", point)
            expect(picker).to_have_value(str(index))
            times.setdefault("tooltip_to_frame_ms", []).append(round(delay, 1))
    return times


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="URL of a disposable notebook in a local JupyterLab with --expose-app-in-browser")
    parser.add_argument("--repetitions", default=3, type=int)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chromium")
        try:
            page = browser.new_page(viewport={"width": 1366, "height": 768})
            print(json.dumps({"engine": browser.version, **measure(page, args.url, args.repetitions)}, indent=2))
        finally:
            browser.close()
