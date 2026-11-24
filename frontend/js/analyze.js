/* HateXAI — detection workspace. */

const EXAMPLES = [
  "The poster said \"immigrants are vermin\" and the audience applauded. I am quoting it here because it should be condemned.",
  "Anyone claiming that community are vermin is repeating a lie. Please stop sharing that meme.",
  "Fuck, the build broke again five minutes before the demo.",
  "We should just make sure our kids go to a school with the right kind of children.",
  "Woh log desh ke liye threat hain, unhe yahan rehne nahi dena chahiye.",
  "Mere dost ne kal interview crack kiya, results aaj aayenge. Wish him luck!",
];

const state = { operating: null, last: null };

function threeWayVerdict(result) {
  const p = result.probability_hate;
  const op = state.operating;
  if (op && typeof op.low === "number" && typeof op.high === "number") {
    if (p >= op.high) return "hate";
    if (p <= op.low) return "non_hate";
    return "review";
  }
  return result.label_id === 1 ? "hate" : "non_hate";
}

const VERDICT_LABEL = { hate: "Hate", non_hate: "Non-hate", review: "Needs review" };

function renderSkeleton() {
  const body = $("#results-body");
  clear(body);
  body.append(
    el("div", {}, [
      el("div", { class: "skeleton", style: "width:40%" }),
      el("div", { class: "skeleton", style: "width:80%" }),
      el("div", { class: "skeleton", style: "width:65%" }),
      el("div", { class: "skeleton", style: "height:70px" }),
      el("p", { class: "muted small", text: "Scoring and attributing tokens…" }),
    ])
  );
}

function renderTokens(tokens, scores) {
  const max = Math.max(1e-6, ...scores.map((s) => Math.abs(s)));
  return el(
    "div",
    { class: "tokens" },
    tokens.map((token, i) => {
      const score = scores[i] || 0;
      const magnitude = Math.abs(score) / max;
      const cls = score > 0.02 * max ? "pos" : score < -0.02 * max ? "neg" : "neutral";
      const node = el("span", {
        class: `tok ${cls}`,
        text: token,
        title: `${score >= 0 ? "+" : ""}${score.toFixed(4)} (${score > 0 ? "pushes toward hate" : "pushes toward non-hate"})`,
        style: cls === "neutral" ? "" : `opacity:${(0.45 + 0.55 * magnitude).toFixed(2)}`,
      });
      return node;
    })
  );
}

function renderResult(result) {
  const body = clear($("#results-body"));
  const verdict = threeWayVerdict(result);
  const exp = result.explanation;

  body.append(
    el("div", { class: "result-head" }, [
      el("span", { class: `verdict ${verdict}`, text: VERDICT_LABEL[verdict] }),
      el("span", { class: "muted small", text: `language: ${result.language}${result.code_mixed ? " · code-mixed" : ""}` }),
    ])
  );

  // Probability bars
  const bars = el("div", { class: "prob-bars" });
  [
    ["hate", result.probability_hate, "hate"],
    ["non-hate", result.probability_non_hate, "non_hate"],
  ].forEach(([label, value, cls]) => {
    const fill = el("div", { class: `prob-fill ${cls}` });
    bars.append(
      el("div", { class: "prob-row" }, [
        el("span", { class: "prob-label", text: label }),
        el("div", { class: "prob-track" }, [fill]),
        el("span", { class: "prob-value", text: pct(value) }),
      ])
    );
    requestAnimationFrame(() => (fill.style.width = `${Math.max(1, value * 100)}%`));
  });
  body.append(bars);

  // Uncertainty + faithfulness tiles
  const unc = result.uncertainty || {};
  body.append(
    el("div", { class: `uncertainty ${unc.is_uncertain ? "warn" : "ok"}` }, [
      el("strong", { text: unc.is_uncertain ? "⚠ Uncertain" : "✓ Confident" }),
      el("span", {
        class: "small muted",
        text: `margin ${num(unc.margin, 3)} · entropy ${num(unc.normalized_entropy, 3)}${
          unc.is_uncertain ? " — escalate to human review" : ""
        }`,
      }),
    ])
  );

  const faith = exp.faithfulness || {};
  body.append(
    el("div", { class: "grid grid-3", style: "margin-top:16px" }, [
      el("div", { class: "metric-tile" }, [
        el("div", { class: "k", text: "Confidence (calibrated)" }),
        el("div", { class: "v", text: pct(result.confidence) }),
      ]),
      el("div", { class: "metric-tile" }, [
        el("div", { class: "k", text: "Comprehensiveness" }),
        el("div", { class: "v", text: num(faith.comprehensiveness, 3) }),
      ]),
      el("div", { class: "metric-tile" }, [
        el("div", { class: "k", text: "Sufficiency" }),
        el("div", { class: "v", text: num(faith.sufficiency, 3) }),
      ]),
    ])
  );

  // Tokens
  body.append(el("h3", { style: "margin-top:22px", text: `Influential tokens (${exp.method})` }));
  body.append(renderTokens(exp.tokens || [], exp.scores || []));
  body.append(
    el("p", { class: "muted small", text: "Red pushes toward hate; green pushes toward non-hate. Opacity reflects attribution magnitude." })
  );

  // Evidence + rationale (kept visually separate: evidence is model-derived)
  body.append(el("h3", { style: "margin-top:20px", text: "Model-derived evidence" }));
  body.append(
    el("div", { class: "chips" }, (exp.top_evidence || []).map((t) => el("span", { class: "chip", text: t })))
  );
  body.append(el("h3", { style: "margin-top:20px", text: "Generated rationale" }));
  body.append(el("div", { class: "rationale", text: exp.rationale }));
  body.append(el("div", { class: "disclaimer", style: "margin-top:12px", text: exp.disclaimer }));

  showAlert($("#form-alert"), "ok", "Analysis complete.");
  $("#report-btn").classList.remove("hidden");
  state.last = result;
}

async function analyze(event) {
  event.preventDefault();
  const text = $("#post-text").value;
  const context = $("#post-context").value;
  const method = $("#method").value;
  const alertBox = $("#form-alert");

  if (!text.trim()) {
    showAlert(alertBox, "error", "Please enter some text to analyze.");
    $("#post-text").focus();
    return;
  }

  const button = $("#analyze-btn");
  button.disabled = true;
  clear(button);
  button.append(el("span", { class: "spinner" }), document.createTextNode(" Analyzing…"));
  clear(alertBox);
  renderSkeleton();

  try {
    const result = await API.post("/api/explain", { text, context, method });
    renderResult(result);
  } catch (err) {
    clear($("#results-body"));
    $("#results-body").append(el("div", { class: "empty", text: "Analysis failed." }));
    showAlert(alertBox, "error", err.message);
    $("#report-btn").classList.add("hidden");
  } finally {
    button.disabled = false;
    clear(button);
    button.textContent = "Analyze";
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  const textarea = $("#post-text");
  const counter = $("#char-counter");
  const updateCounter = () => {
    const len = textarea.value.length;
    counter.textContent = `${len} / 5000`;
    counter.classList.toggle("over", len > 4800);
  };
  textarea.addEventListener("input", updateCounter);
  updateCounter();

  const chips = $("#examples");
  EXAMPLES.forEach((example) => {
    chips.append(
      el("button", {
        class: "chip",
        type: "button",
        text: example.length > 46 ? `${example.slice(0, 44)}…` : example,
        title: example,
        onclick: () => {
          textarea.value = example;
          updateCounter();
          textarea.focus();
        },
      })
    );
  });

  const prefill = sessionStorage.getItem("hatexai.prefill");
  if (prefill) {
    textarea.value = prefill;
    updateCounter();
    sessionStorage.removeItem("hatexai.prefill");
  }

  $("#analyze-form").addEventListener("submit", analyze);
  $("#reset-btn").addEventListener("click", () => {
    clear($("#results-body"));
    $("#results-body").append(el("div", { class: "empty", text: "No analysis yet. Submit a post to see the decision and explanation." }));
    clear($("#form-alert"));
    $("#report-btn").classList.add("hidden");
    updateCounter();
  });

  $("#report-btn").addEventListener("click", () => {
    if (!state.last) return;
    sessionStorage.setItem(
      "hatexai.feedback",
      JSON.stringify({
        text: textarea.value,
        context: $("#post-context").value,
        predicted_label: state.last.label,
        probability_hate: state.last.probability_hate,
        model_version: state.last.model ? state.last.model.version : "",
      })
    );
    window.location.href = "feedback.html";
  });

  // Operating point drives the three-way verdict; if it is missing, we fall
  // back to the raw label rather than inventing a threshold.
  try {
    const evaluation = await API.get("/api/evaluation");
    state.operating = (evaluation.available && evaluation.available.threshold) || null;
  } catch {
    state.operating = null;
  }
});
