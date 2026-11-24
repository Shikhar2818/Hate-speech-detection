/* HateXAI — evaluation dashboard. */

function barRow(label, value, opts = {}) {
  const row = el("div", { class: "bar-row" }, [
    el("span", { class: "muted", text: label, title: label }),
    el("div", { class: "bar-track" }, [el("div", { class: `bar-fill ${opts.cls || ""}` })]),
    el("span", { class: "mono small", text: value === null || value === undefined ? "n/a" : pct(value) }),
  ]);
  const fill = row.querySelector(".bar-fill");
  requestAnimationFrame(() => (fill.style.width = `${Math.max(0, Math.min(1, value || 0)) * 100}%`));
  return row;
}

function metricTile(key, value) {
  return el("div", { class: "metric-tile" }, [
    el("div", { class: "k", text: key }),
    el("div", { class: "v", text: value }),
  ]);
}

function sectionOrEmpty(container, hasData, emptyMessage) {
  if (!hasData) {
    clear(container);
    container.append(el("div", { class: "empty", text: emptyMessage }));
    return false;
  }
  return true;
}

function renderManifest(manifest) {
  const node = clear($("#manifest"));
  if (!manifest) {
    node.append(el("div", { class: "empty", text: "No run manifest found. Run the experiment script to generate artifacts." }));
    return;
  }
  node.append(
    el("div", { class: "card-head" }, [el("h2", { style: "margin:0", text: "Run manifest" })]),
    el("div", { class: "grid grid-4" }, [
      metricTile("Generated", manifest.generated_at ? manifest.generated_at.slice(0, 19).replace("T", " ") : "n/a"),
      metricTile("Seed", String(manifest.seed ?? "n/a")),
      metricTile("Device", manifest.device || "n/a"),
      metricTile("Python", manifest.python || "n/a"),
    ])
  );
}

function renderHeadline(test) {
  const container = clear($("#headline-metrics"));
  if (!test || test.unavailable) {
    container.append(el("div", { class: "empty", text: "No test metrics available." }));
    return;
  }
  $("#headline-scope").textContent = `test split · n = ${test.n}`;
  container.append(
    metricTile("Accuracy", num(test.accuracy, 3)),
    metricTile("Macro-F1", num(test.f1_macro, 3)),
    metricTile("AUROC", num(test.auroc, 3)),
    metricTile("ECE (calibration)", num(test.ece, 3))
  );

  const perClass = clear($("#per-class"));
  Object.entries(test.per_class || {}).forEach(([label, m]) => {
    const name = label === "1" ? "hate" : "non-hate";
    perClass.append(barRow(`${name} (n=${m.support})`, m.f1));
  });

  const rates = clear($("#error-rates"));
  rates.append(
    barRow("False positive rate", test.false_positive_rate, { cls: "warn" }),
    barRow("False negative rate", test.false_negative_rate, { cls: "danger" })
  );

  const cm = clear($("#confusion"));
  const matrix = test.confusion_matrix || {};
  [
    ["tp", "True positive", matrix.tp],
    ["fp", "False positive", matrix.fp],
    ["fn", "False negative", matrix.fn],
    ["tn", "True negative", matrix.tn],
  ].forEach(([cls, label, value]) => {
    cm.append(el("div", { class: `cm-cell ${cls}` }, [el("div", { class: "n", text: String(value ?? "–") }), el("div", { class: "l", text: label })]));
  });
}

function renderThreshold(threshold, threeWay) {
  const note = $("#threshold-note");
  const metrics = clear($("#threshold-metrics"));
  if (!threshold) {
    note.textContent = "No operating point selected.";
    metrics.append(el("div", { class: "empty", text: "Threshold unavailable." }));
    return;
  }
  note.textContent = threshold.note || "";
  metrics.append(
    metricTile("Threshold", num(threshold.threshold, 3)),
    metricTile("Review band", `${num(threshold.low, 2)} – ${num(threshold.high, 2)}`),
    metricTile("FN : FP cost", `${threshold.fn_cost} : ${threshold.fp_cost}`),
    metricTile("Degenerate", threshold.degenerate ? "yes" : "no")
  );

  const tw = clear($("#three-way"));
  if (threeWay) {
    const decided = threeWay.decided || 0;
    const abstained = threeWay.abstained || 0;
    const total = Math.max(1, threeWay.n || decided + abstained);
    tw.append(barRow("Decided", decided / total, { cls: "accent" }), barRow("Needs human review", abstained / total, { cls: "warn" }));
  } else {
    tw.append(el("div", { class: "empty", text: "No three-way results." }));
  }

  const op = clear($("#operating"));
  if (threeWay && threeWay.decided_metrics && !threeWay.decided_metrics.unavailable) {
    const m = threeWay.decided_metrics;
    op.append(
      el("div", { class: "grid grid-2" }, [
        metricTile("Decided accuracy", num(m.accuracy, 3)),
        metricTile("Decided macro-F1", num(m.f1_macro, 3)),
      ]),
      el("p", { class: "muted small", style: "margin-top:10px", text: "Metrics computed over non-abstained examples only; abstentions are counted separately." })
    );
  } else {
    op.append(
      el("div", { class: "empty", text: (threeWay && threeWay.decided_metrics && threeWay.decided_metrics.unavailable) || "No decided examples." })
    );
  }
}

function renderComparison(comparison) {
  const body = clear($("#comparison-body"));
  if (!comparison) {
    body.append(el("tr", {}, [el("td", { colspan: "7", text: "No comparison artifact." })]));
    return;
  }
  comparison.systems.forEach((s) => {
    const status = s.available
      ? el("span", { class: "pill ok", text: "evaluated" })
      : el("span", { class: "pill neutral", text: "not run" });
    body.append(
      el("tr", {}, [
        el("td", { text: s.name }),
        el("td", {}, [status]),
        el("td", { class: "num", text: s.accuracy === null ? "–" : num(s.accuracy, 3) }),
        el("td", { class: "num", text: s.f1_macro === null ? "–" : num(s.f1_macro, 3) }),
        el("td", { class: "num", text: s.auroc === null ? "–" : num(s.auroc, 3) }),
        el("td", { class: "num", text: s.ece === null ? "–" : num(s.ece, 3) }),
        el("td", { class: "muted small", text: s.note }),
      ])
    );
  });
  $("#comparison-caveat").textContent = comparison.caveat || "";
}

function renderAblation(ablation) {
  const body = clear($("#ablation-body"));
  if (!ablation) {
    body.append(el("tr", {}, [el("td", { colspan: "8", text: "No ablation artifact." })]));
    return;
  }
  ablation.rows.forEach((r) => {
    const selected = ablation.selected && r.feature_set === ablation.selected.feature_set && r.calibrated === ablation.selected.calibrate;
    body.append(
      el("tr", { style: selected ? "background: rgba(139,124,246,.12)" : "" }, [
        el("td", {}, [el("span", { text: r.feature_set }), selected ? el("span", { class: "pill ok", style: "margin-left:8px", text: "selected" }) : document.createTextNode("")]),
        el("td", { text: r.calibrated ? "yes" : "no" }),
        el("td", { class: "num", text: r.cv_f1_macro_train === null ? "n/a" : num(r.cv_f1_macro_train, 3) }),
        el("td", { class: "num", text: num(r.test_accuracy, 3) }),
        el("td", { class: "num", text: num(r.test_brier, 3) }),
        el("td", { class: "num", text: num(r.test_ece, 3) }),
        el("td", { class: "num", text: num(r.probe_overall_accuracy, 3) }),
        el("td", { class: "num", text: num(r.probe_unambiguous_accuracy, 3) }),
      ])
    );
  });
  $("#ablation-note").textContent = ablation.note || "";
}

function renderProbes(probes) {
  if (!probes || !probes.by_category) {
    $("#probe-categories").append(el("div", { class: "empty", text: "No probe results." }));
    return;
  }
  const cats = clear($("#probe-categories"));
  Object.entries(probes.by_category).forEach(([name, info]) => {
    if (info.accuracy === null) {
      cats.append(el("div", { class: "bar-row" }, [el("span", { class: "muted", text: name }), el("span", { class: "muted small", text: `insufficient support (n=${info.support})` })]));
    } else {
      cats.append(barRow(`${name} (n=${info.support})`, info.accuracy, { cls: info.accuracy < 0.5 ? "danger" : "accent" }));
    }
  });

  const subsets = clear($("#probe-subsets"));
  [
    ["All probes", probes.overall],
    ["Unambiguous subset", probes.unambiguous],
    ["Contested subset", probes.ambiguous],
  ].forEach(([label, m]) => {
    if (m && m.unavailable) {
      subsets.append(el("div", { class: "bar-row" }, [el("span", { class: "muted", text: label }), el("span", { class: "muted small", text: m.unavailable })]));
    } else if (m) {
      subsets.append(barRow(`${label} (n=${m.n})`, m.accuracy, { cls: m.accuracy < 0.5 ? "danger" : "accent" }));
    }
  });

  const failures = clear($("#probe-failures"));
  $("#probe-note").textContent = `Threshold ${num(probes.threshold, 3)}: ${(probes.failures || []).length} of ${probes.n} probes misclassified. Author-constructed probe set — a diagnostic signal, not a benchmark.`;
  if (!probes.failures || !probes.failures.length) {
    failures.append(el("div", { class: "empty", text: "No misclassified probes." }));
    return;
  }
  const table = el("table", {}, [
    el("thead", {}, [el("tr", {}, [el("th", { text: "ID" }), el("th", { text: "Category" }), el("th", { text: "Gold" }), el("th", { text: "Predicted" }), el("th", { class: "num", text: "P(hate)" }), el("th", { text: "Text" })])]),
    el("tbody", {}, probes.failures.slice(0, 25).map((f) =>
      el("tr", {}, [
        el("td", { class: "mono small", text: f.id }),
        el("td", {}, [el("span", { class: "pill neutral", text: f.category }), f.ambiguous ? el("span", { class: "pill warn", style: "margin-left:6px", text: "contested" }) : document.createTextNode("")]),
        el("td", { text: f.gold === 1 ? "hate" : "non-hate" }),
        el("td", { text: f.pred === 1 ? "hate" : "non-hate" }),
        el("td", { class: "num", text: num(f.prob_hate, 3) }),
        el("td", { class: "muted small", text: f.text.slice(0, 90) }),
      ])
    )),
  ]);
  failures.append(table);
}

function renderFaithfulness(faith) {
  const container = clear($("#faithfulness"));
  if (!faith) {
    container.append(el("div", { class: "empty", text: "No faithfulness artifact." }));
    return;
  }
  container.append(
    metricTile(`Mean comprehensiveness (n=${faith.n})`, num(faith.mean_comprehensiveness, 3)),
    metricTile(`Mean sufficiency (n=${faith.n})`, num(faith.mean_sufficiency, 3))
  );
  $("#faith-note").textContent = faith.note || "";
}

async function loadDashboard() {
  let data;
  try {
    data = await API.get("/api/evaluation");
  } catch (err) {
    showAlert($("#dashboard-alert"), "error", err.message);
    clear($("#manifest"));
    return;
  }
  const available = data.available || {};
  showAlert($("#dashboard-alert"), "ok", "Artifacts loaded from outputs/eval.");
  renderManifest(available.run_manifest);
  renderHeadline(available.baseline_test);
  renderThreshold(available.threshold, available.test_three_way);
  renderComparison(available.comparison);
  renderAblation(available.ablation);
  renderProbes(available.probes);
  renderFaithfulness(available.faithfulness);
}

document.addEventListener("DOMContentLoaded", loadDashboard);
