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
  const abort = new AbortController();
  if (model.get("fullscreen") && el.ownerDocument.fullscreenEnabled) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "pg-fullscreen";
    button.textContent = "Toàn màn hình";
    const doc = el.ownerDocument;
    let active = false;
    function syncFullscreen() {
      const wasActive = active;
      active = doc.fullscreenElement === el.closest(".penguin-panel");
      button.textContent = active ? "Thoát toàn màn hình" : "Toàn màn hình";
      if (wasActive && !active) button.focus();
    }
    button.addEventListener("click", async () => {
      try {
        if (active) await doc.exitFullscreen();
        else await el.closest(".penguin-panel").requestFullscreen();
      } catch {
        button.title = "Trình duyệt chưa cho phép toàn màn hình trong cửa sổ này.";
      }
    }, {signal:abort.signal});
    doc.addEventListener("fullscreenchange", syncFullscreen, {signal:abort.signal});
    el.append(button);
  }
  return () => { abort.abort(); model.off("change:value", sync); };
}
export default { render };
