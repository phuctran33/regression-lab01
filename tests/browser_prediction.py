"""Optional browser acceptance check for section 5 in a running local JupyterLab."""
import argparse
from pathlib import Path
import re

from playwright.sync_api import expect, sync_playwright


def exercise(page, url, axe_path):
    page.goto(url)
    page.wait_for_function("""() => {
        const w = window.jupyterapp?.shell.currentWidget;
        return w?.context?.path?.endsWith('.ipynb') &&
               w.sessionContext?.session?.kernel?.connectionStatus === 'connected';
    }""", timeout=120_000)
    page.evaluate("""async () => {
        const app = window.jupyterapp, w = app.shell.currentWidget;
        await w.sessionContext.ready;
        await w.sessionContext.session.kernel.info;
        app.shell.collapseLeft();
        await app.commands.execute('notebook:run-all-cells');
        const i = w.content.widgets.findIndex(cell => cell.model.sharedModel.getSource().includes('display(prediction_app.panel)'));
        if (i < 0) throw new Error('Missing prediction application cell');
        await w.content.scrollToItem(i);
    }""")
    app = page.get_by_role("region", name="Dự đoán khối lượng chim cánh cụt")
    expect(app.locator(".pg-mass")).to_have_text("3.571,7 g", timeout=120_000)
    expect(app.locator(".pg-bounds")).to_have_text("2.935,5 g đến 4.208,0 g")
    page.add_script_tag(path=str(axe_path))

    def audit(state):
        findings = app.evaluate("""async node => {
            const tags = ['wcag2a','wcag2aa','wcag21a','wcag21aa','wcag22aa'];
            const r = await axe.run(node, {runOnly:{type:'tag',values:tags}});
            return {violations:r.violations.map(x=>x.id), incomplete:r.incomplete.map(x=>x.id)};
        }""")
        print(state, findings)
        assert not findings["violations"], findings

    audit("desktop")
    field = app.get_by_role("spinbutton").first
    button = app.get_by_role("button", name="Dự đoán", exact=True)
    field.fill("")
    expect(button).to_be_disabled()
    expect(app.locator(".pg-answer")).to_have_count(0)
    field.fill("197")
    expect(button).to_be_enabled()
    button.click()
    expect(app.locator(".pg-mass")).to_have_text("3.571,7 g")
    page.set_viewport_size({"width": 390, "height": 844})
    app.get_by_role("button", name="Xem biểu đồ").click()
    trend = app.get_by_role("radio", name="Xu hướng")
    expect(trend).to_be_focused()
    trend.press("ArrowRight")
    img = app.locator(".pg-chart img")
    expect(img).to_have_attribute("alt", re.compile("Histogram"))
    expect(img).to_have_attribute("data-layout", "compact")
    app.get_by_text("Xem số liệu của biểu đồ", exact=True).click()
    counts = app.locator(".pg-caption tbody tr td:last-child").all_inner_texts()
    assert sum(map(int, counts)) == 69, counts
    audit("mobile histogram")
    assert app.evaluate("node => node.scrollWidth <= node.clientWidth + 1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="Local JupyterLab URL for a disposable copy of lab01.ipynb")
    parser.add_argument("--axe", type=Path, required=True, help="Local axe-core distribution (axe.min.js)")
    parser.add_argument("--browser", choices=("chromium", "firefox", "webkit"), default="chromium")
    args = parser.parse_args()
    with sync_playwright() as playwright:
        browser = (playwright.chromium.launch(channel="chromium") if args.browser == "chromium"
                   else getattr(playwright, args.browser).launch())
        try:
            print(args.browser, browser.version)
            exercise(browser.new_page(viewport={"width": 1440, "height": 900}), args.url, args.axe)
        finally:
            browser.close()
