"use strict";

const picker = document.getElementById("run");
const panels = Array.from(document.querySelectorAll(".run"));

function selectRun(run) {
  picker.value = run.id;
  for (const panel of panels) {
    panel.hidden = panel !== run;
  }
}

function selectFromHash() {
  const hash = window.location.hash.slice(1);
  if (!hash) {
    selectRun(panels[0]);
    return;
  }
  const run = panels.find(panel => encodeURIComponent(panel.id) === hash || panel.dataset.alias === hash);
  if (run) {
    selectRun(run);
    run.scrollIntoView();
  }
}

if (panels.length) {
  selectRun(panels[0]);
  picker.disabled = false;
  picker.addEventListener("change", () => {
    const run = document.getElementById(picker.value);
    selectRun(run);
    window.history.pushState(null, "", `#${encodeURIComponent(run.id)}`);
  });
  window.addEventListener("hashchange", selectFromHash);
  selectFromHash();
}
