const fieldNotes = {
  flipper_length_mm: "Từ vai đến chóp cánh bơi.",
  bill_length_mm: "Từ chân phần mỏ lộ ra đến chóp, dọc sống mỏ trên.",
  bill_depth_mm: "Khoảng theo chiều đứng giữa mặt trên và mặt dưới mỏ."
};
const speciesNames = {Adelie:"Pygoscelis adeliae", Chinstrap:"Pygoscelis antarcticus", Gentoo:"Pygoscelis papua"};
function render({model, el}) {
  el.classList.add("pg-atlas");
  el.innerHTML = '<figure><figcaption><strong></strong> <i></i></figcaption>' + model.get("svg") +
    '<p class="pg-anatomy-note"></p><small>Sơ đồ vị trí đo · không theo tỷ lệ</small></figure>';
  const panel = () => el.closest(".penguin-panel");
  const abort = new AbortController();
  let active = "flipper_length_mm";
  function update() {
    const species = model.get("species");
    el.dataset.species = species;
    el.querySelector("strong").textContent = species;
    el.querySelector("i").textContent = speciesNames[species] || "";
    for (const mark of el.querySelectorAll("[data-measure]")) {
      mark.classList.toggle("pg-chosen", mark.dataset.measure === active);
    }
    const value = model.get("values")[active];
    el.querySelector(".pg-anatomy-note").textContent = fieldNotes[active] +
      (value == null ? " Số đo cần kiểm tra." : " " + value.toLocaleString("vi-VN") + " mm.");
  }
  function identify(e) {
    const marker = e.target.closest("[data-measure], [data-field]");
    const key = marker?.dataset.measure || marker?.dataset.field;
    if (fieldNotes[key] && active !== key) { active = key; update(); }
  }
  function edit(e) {
    const mark = e.target.closest("[data-measure]");
    if (!mark) return;
    if (e.type === "keydown") {
      if (!["Enter"," "].includes(e.key)) return;
      e.preventDefault(); e.stopPropagation();
    }
    panel()?.querySelector('[data-field="'+mark.dataset.measure+'"] input[type=number]')?.focus();
  }
  for (const type of ["focusin","pointerover","pointerdown"])
    el.ownerDocument.addEventListener(type, e => { if (panel()?.contains(e.target)) identify(e); }, {signal:abort.signal});
  el.addEventListener("click", edit, {signal:abort.signal});
  el.addEventListener("keydown", edit, {signal:abort.signal});
  model.on("change",update); update();
  return () => { abort.abort(); model.off("change",update); };
}
export default {render};
