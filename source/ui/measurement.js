// A single measurement shared by a native range and a directly editable number.
function render({ model, el }) {
  el.setAttribute("data-lm-suppress-shortcuts", "true");
  // Let the native fields handle keys before the notebook handles cell navigation.
  el.addEventListener("keydown", (event) => {
    if (!event.metaKey && !event.ctrlKey && !event.altKey && event.key !== "Escape") {
      event.stopPropagation();
    }
  });
  const label = document.createElement("label");
  const number = document.createElement("input");
  const range = document.createElement("input");
  const limits = document.createElement("div");
  const hint = document.createElement("span");
  const title = model.get("description");
  const id = "measure-" + crypto.getRandomValues(new Uint32Array(4)).join("-");
  el.classList.add("pg-measure");
  el.lang = "vi";
  const format = new Intl.NumberFormat("vi-VN");
  label.htmlFor = id;
  label.textContent = title;
  number.type = "number";
  number.id = id;
  number.required = true;
  range.type = "range";
  range.setAttribute("aria-label", title + " · thanh trượt");
  limits.className = "pg-limits";
  limits.setAttribute("aria-hidden", "true");
  hint.id = id + "-hint";
  hint.className = "pg-field-error";
  number.setAttribute("aria-describedby", hint.id);
  for (const input of [number, range]) {
    for (const attr of ["min", "max", "step"]) input[attr] = model.get(attr);
  }
  for (const bound of [range.min, range.max]) {
    const span = document.createElement("span");
    span.textContent = format.format(+bound) + " mm";
    limits.append(span);
  }
  function syncValue() {
    range.value = model.get("value");
    if (document.activeElement !== number || number.valueAsNumber !== +range.value) {
      number.value = range.value;
    }
  }
  function update(input) {
    const valid = input.validity.valid && Number.isFinite(input.valueAsNumber);
    hint.textContent = valid ? "" : `Nhập từ ${format.format(+range.min)} đến ${format.format(+range.max)} mm, bước ${format.format(+range.step)} mm.`;
    number.setAttribute("aria-invalid", String(!valid));
    model.set("valid", valid);
    if (valid) {
      if (input === range) number.value = range.value;
      else range.value = number.value;
      model.set("value", input.valueAsNumber);
    }
    model.save_changes();
  }
  range.addEventListener("input", () => update(range));
  number.addEventListener("input", () => update(number));
  model.on("change:value", syncValue);
  function reset() {
    number.value = model.get("value");
    range.value = number.value;
    number.removeAttribute("aria-invalid");
    hint.textContent = "";
  }
  model.on("change:reset_count", reset);
  syncValue();
  el.append(label, number, range, limits, hint);
  return () => { model.off("change:value", syncValue); model.off("change:reset_count", reset); };
}
export default { render };
