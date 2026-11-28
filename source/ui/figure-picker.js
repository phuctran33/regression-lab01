// One native radio group per notebook output; arrow keys select a figure.
function render({ model, el }) {
  const group = document.createElement("fieldset");
  const legend = document.createElement("legend");
  legend.textContent = model.get("legend");
  legend.className = "pg-sr-only";
  group.className = "pg-views";
  group.lang = "vi";
  group.append(legend);
  const name = "figure-" + crypto.getRandomValues(new Uint32Array(4)).join("-");
  for (const [title, value] of model.get("options")) {
    const label = document.createElement("label");
    const radio = document.createElement("input");
    radio.type = "radio";
    radio.name = name;
    radio.value = value;
    label.append(radio, document.createTextNode(title));
    group.append(label);
  }
  group.addEventListener("keydown", (event) => {
    if (["ArrowRight", "ArrowLeft", "ArrowUp", "ArrowDown", " "].includes(event.key)) event.stopPropagation();
  });
  group.addEventListener("change", (event) => {
    model.set("value", event.target.value);
    model.save_changes();
  });
  function sync() {
    for (const radio of group.querySelectorAll("input")) radio.checked = radio.value === model.get("value");
  }
  model.on("change:value", sync);
  sync();
  el.append(group);
  return () => model.off("change:value", sync);
}
export default { render };
