function render({ model, el }) {
  el.setAttribute("data-lm-suppress-shortcuts", "true");
  el.addEventListener("keydown", (event) => {
    if (event.target.matches("summary, .pg-jump") && [" ", "Enter"].includes(event.key)) event.stopPropagation();
  });
  const abort = new AbortController();
  function invalidateExport(event) {
    if (!event.target.closest(".pg-form select,.pg-form input,.pg-command select")) return;
    if (!el.closest(".penguin-panel")?.contains(event.target)) return;
    const link = el.querySelector(".pg-export");
    if (!link) return;
    if (event.target.closest(".pg-preset") && event.target.selectedIndex === 0) return;
    const field = event.target.closest("[data-field]")?.dataset.field;
    if (field && event.target.validity.valid &&
        event.target.valueAsNumber === JSON.parse(link.dataset.measurements)[field]) return;
    link.remove();
  }
  el.ownerDocument.addEventListener("input", invalidateExport, {capture:true, signal:abort.signal});
  const output = document.createElement("div");
  const status = document.createElement("div");
  status.className = "pg-sr-only";
  status.setAttribute("role", "status");
  status.setAttribute("aria-atomic", "true");
  el.append(output, status);
  output.addEventListener("click", (event) => {
    if (event.target.closest(".pg-edit")) {
      const radio = el.closest(".penguin-panel").querySelector('.pg-nav input[value="predict"]');
      radio.click();
      radio.focus();
      const form = el.closest(".penguin-panel").querySelector(".pg-form");
      const focusField = () => {
        if (!form.getClientRects().length || !radio.checked) return;
        observer.disconnect();
        if (document.activeElement === radio) form.querySelector("select")?.focus();
      };
      const observer = new ResizeObserver(focusField);
      observer.observe(form);
      focusField();
      setTimeout(() => observer.disconnect(), 3000);
      return;
    }
    const jump = event.target.closest(".pg-jump");
    if (!jump) return;
    const targets = {chart: ".pg-view .pg-views input:checked", predict: ".pg-command button:not(.pg-reset)",
                     result: ".pg-answer h3"};
    const panel = el.closest(".penguin-panel");
    const visible = [...panel.querySelectorAll(targets[jump.dataset.target])].find(node => node.getClientRects().length);
    (visible || panel.querySelector(".pg-view h3"))?.focus();
  });
  function show() {
    // Editing a measurement must not move the form underneath the pointer.
    const previous = output.querySelector(".pg-answer, .pg-notice");
    const height = previous ? output.getBoundingClientRect().height : 0;
    const wasOpen = output.querySelector("details")?.open;
    output.innerHTML = model.get("value");
    const notice = output.querySelector(".pg-notice");
    if (notice && height) notice.style.minHeight = height + "px";
    const details = output.querySelector("details");
    if (details && wasOpen) details.open = true;
  }
  function announce() { status.textContent = model.get("announcement"); }
  function closeDetails() {
    for (const details of output.querySelectorAll("details[open]")) details.open = false;
  }
  model.on("change:value", show);
  model.on("change:announcement", announce);
  model.on("change:reset_count", closeDetails);
  show();
  announce();
  return () => {
    abort.abort();
    model.off("change:value", show);
    model.off("change:announcement", announce);
    model.off("change:reset_count", closeDetails);
  };
}
export default { render };
