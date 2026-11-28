function render({ model, el }) {
  el.setAttribute("data-lm-suppress-shortcuts", "true");
  el.addEventListener("keydown", (event) => {
    if (event.target.matches("summary") && [" ", "Enter"].includes(event.key)) event.stopPropagation();
  });
  const output = document.createElement("div");
  const status = document.createElement("div");
  status.className = "pg-sr-only";
  status.setAttribute("role", "status");
  status.setAttribute("aria-atomic", "true");
  el.append(output, status);
  function show() { output.innerHTML = model.get("value"); }
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
