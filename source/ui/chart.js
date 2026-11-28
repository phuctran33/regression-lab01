// The two sets of normalized mark positions come from the rendered Matplotlib axes.
function render({ model, el }) {
  const frame = document.createElement("div");
  frame.className = "pg-inspector";
  frame.lang = "vi";
  frame.innerHTML = `
    <div class="pg-plot-stage"><img alt=""><span class="pg-halo" aria-hidden="true"></span>
      <span class="pg-point-tag" aria-hidden="true" hidden></span></div>
    <div class="pg-inspect-tools"><div class="pg-picker-field"><label>Điểm dữ liệu</label><select></select></div>
      <button type="button" class="pg-expand">Mở rộng biểu đồ</button></div>
    <p class="pg-tip"></p><span class="pg-sr-only" role="status" aria-live="polite"></span>`;
  const dialog = document.createElement("dialog");
  dialog.className = "pg-chart-dialog";
  dialog.setAttribute("aria-label", "Biểu đồ mở rộng");
  const close = document.createElement("button");
  close.type = "button";
  close.textContent = "Đóng biểu đồ";
  const heading = document.createElement("h3");
  heading.id = "pg-focus-" + crypto.getRandomValues(new Uint32Array(4)).join("-");
  const header = document.createElement("header");
  header.className = "pg-dialog-head";
  header.append(heading, close);
  dialog.setAttribute("aria-describedby", heading.id);
  dialog.append(header);
  el.append(frame, dialog);
  const stage = frame.querySelector(".pg-plot-stage");
  const img = stage.querySelector("img");
  const halo = stage.querySelector(".pg-halo");
  const tag = stage.querySelector(".pg-point-tag");
  const picker = frame.querySelector("select");
  picker.id = "pg-point-" + crypto.getRandomValues(new Uint32Array(4)).join("-");
  frame.querySelector("label").htmlFor = picker.id;
  const expand = frame.querySelector(".pg-expand");
  const tip = frame.querySelector(".pg-tip");
  const status = frame.querySelector("[role=status]");
  let payload, currentBytes, url, positions = [], index = 0;
  function locate(i, announce = false, reveal = false) {
    if (!positions.length || !payload.labels[i]) return;
    index = i;
    picker.value = String(i);
    const [x, y] = positions[i];
    halo.style.left = (100*x) + "%";
    halo.style.top = (100*y) + "%";
    const [title, detail] = payload.labels[i];
    if (reveal) tag.hidden = false;
    tag.textContent = title;
    tag.style.left = (100*x) + "%";
    tag.style.top = (100*y) + "%";
    tag.dataset.side = x > .5 ? "left" : "right";
    tag.dataset.below = String(y < .15);
    tip.textContent = detail;
    if (announce) status.textContent = title + ". " + detail;
  }
  function update() {
    const panel = el.closest(".penguin-panel");
    if (panel) {
      panel.lang = "vi";
      panel.setAttribute("role", "region");
      panel.setAttribute("aria-label", "Dự đoán khối lượng chim cánh cụt");
    }
    const wide = model.get("value");
    if (!wide.byteLength && dialog.open) dialog.close();
    frame.hidden = !wide.byteLength;
    img.hidden = !wide.byteLength;
    const compact = stage.clientWidth < 440;
    const layout = compact ? "compact" : "wide";
    img.alt = model.get("description");
    img.dataset.layout = layout;
    const next = model.get("inspection");
    if (payload !== next) {
      payload = next;
      index = payload.selected || 0;
      const options = (payload.labels || []).map(([title], i) => new Option((i+1) + ". " + title, i));
      picker.replaceChildren(...options);
      tag.hidden = true;
      tip.textContent = status.textContent = "";
    }
    positions = payload[layout] || [];
    frame.querySelector(".pg-inspect-tools").hidden = !positions.length;
    halo.hidden = !positions.length;
    const bytes = compact && wide.byteLength ? model.get("compact") : wide;
    if (currentBytes !== bytes) {
      currentBytes = bytes;
      if (url) URL.revokeObjectURL(url);
      url = bytes.byteLength ? URL.createObjectURL(new Blob([bytes], {type:"image/png"})) : null;
      if (url) img.src = url;
      else img.removeAttribute("src");
    }
    locate(index);
  }
  function point(event) {
    const box = stage.getBoundingClientRect();
    const px = event.clientX-box.left, py = event.clientY-box.top;
    let best = -1, distance = event.type === "pointerdown" ? 34 : 23;
    positions.forEach(([x,y], i) => {
      const d = Math.hypot(x*box.width-px, y*box.height-py);
      if (d < distance) { best = i; distance = d; }
    });
    if (best >= 0) locate(best, event.type === "pointerdown", true);
    else tag.hidden = true;
  }
  stage.addEventListener("pointermove", point);
  stage.addEventListener("pointerdown", point);
  stage.addEventListener("pointerleave", () => { tag.hidden = true; });
  picker.addEventListener("change", () => locate(+picker.value, true, true));
  frame.addEventListener("keydown", e => {
    if (e.key === "Escape") tag.hidden = true;
    else e.stopPropagation();
  });
  expand.addEventListener("click", () => {
    heading.textContent = el.closest(".pg-view")?.querySelector("h3")?.textContent || img.alt;
    dialog.showModal();
    close.focus();
    dialog.append(frame);
    expand.hidden = true;
    update();
  });
  close.addEventListener("click", () => dialog.close());
  dialog.addEventListener("close", () => {
    el.insertBefore(frame, dialog);
    expand.hidden = false;
    update();
    if (!frame.hidden) expand.focus();
  });
  const ro = new ResizeObserver(update);
  ro.observe(stage);
  model.on("change", update);
  update();
  return () => { ro.disconnect(); model.off("change", update); if (url) URL.revokeObjectURL(url); };
}
export default { render };
