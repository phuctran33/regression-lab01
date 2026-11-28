// The notebook output can be much narrower than the browser viewport.
function render({ model, el }) {
  const img = document.createElement("img");
  el.append(img);
  let currentBytes;
  let url;
  function update() {
    const panel = el.closest(".penguin-panel");
    if (panel && panel.lang !== "vi") {
      panel.lang = "vi";
      panel.setAttribute("role", "region");
      panel.setAttribute("aria-label", "Dự đoán khối lượng chim cánh cụt");
    }
    const wide = model.get("value");
    const compact = el.clientWidth < 440;
    const bytes = compact && wide.byteLength ? model.get("compact") : wide;
    img.alt = model.get("description");
    img.dataset.layout = compact ? "compact" : "wide";
    if (currentBytes === bytes) return;
    currentBytes = bytes;
    if (url) URL.revokeObjectURL(url);
    img.hidden = !bytes.byteLength;
    if (bytes.byteLength) {
      url = URL.createObjectURL(new Blob([bytes], { type: "image/png" }));
      img.src = url;
    } else {
      img.removeAttribute("src");
      url = undefined;
    }
  }
  const observer = new ResizeObserver(update);
  observer.observe(el);
  model.on("change", update);
  update();
  return () => {
    observer.disconnect();
    model.off("change", update);
    if (url) URL.revokeObjectURL(url);
  };
}
export default { render };
