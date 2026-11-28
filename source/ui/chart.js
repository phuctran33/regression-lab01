// Keep the image and its text alternative in the same Jupyter widget view.
function render({ model, el }) {
  const img = document.createElement("img");
  el.append(img);
  let url;
  function update() {
    if (url) URL.revokeObjectURL(url);
    const bytes = model.get("value");
    img.hidden = !bytes.byteLength;
    img.alt = model.get("description");
    if (bytes.byteLength) {
      url = URL.createObjectURL(new Blob([bytes], { type: "image/png" }));
      img.src = url;
    } else {
      img.removeAttribute("src");
      url = undefined;
    }
  }
  model.on("change", update);
  update();
  return () => {
    model.off("change", update);
    if (url) URL.revokeObjectURL(url);
  };
}
export default { render };
