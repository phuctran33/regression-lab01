function render({ model, el }) {
  el.setAttribute("data-lm-suppress-shortcuts", "true");
  el.addEventListener("keydown", (event) => {
    if (event.target.matches("summary, .pg-jump") && [" ", "Enter"].includes(event.key)) event.stopPropagation();
  });
  const output = document.createElement("div");
  const status = document.createElement("div");
  status.className = "pg-sr-only";
  status.setAttribute("role", "status");
  status.setAttribute("aria-atomic", "true");
  el.append(output, status);
  output.addEventListener("click", (event) => {
    const jump = event.target.closest(".pg-jump");
    if (!jump) return;
    const target = jump.dataset.target === "chart" ? ".pg-views input:checked" : ".pg-answer h3";
    el.closest(".penguin-panel")?.querySelector(target)?.focus();
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
  model.on("change:value", show);
  model.on("change:announcement", announce);
  show();
  announce();
  return () => {
    model.off("change:value", show);
    model.off("change:announcement", announce);
  };
}
export default { render };
