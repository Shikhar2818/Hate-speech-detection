/* HateXAI — detection workspace. */

const EXAMPLES = [
  "The poster said \"immigrants are vermin\" and the audience applauded. I am quoting it here because it should be condemned.",
  "Anyone claiming that community are vermin is repeating a lie. Please stop sharing that meme.",
  "Fuck, the build broke again five minutes before the demo.",
  "We should just make sure our kids go to a school with the right kind of children.",
  "Woh log desh ke liye threat hain, unhe yahan rehne nahi dena chahiye.",
  "Mere dost ne kal interview crack kiya, results aaj aayenge. Wish him luck!",
];

const state = { operating: null, last: null, pending: null, memeEnabled: false };

function threeWayVerdict(result) {
  if (result.decision) return result.decision === "needs_review" ? "review" : result.decision;
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
        title: cls === "neutral" ? "Little or no measured influence"
          : `${score >= 0 ? "+" : ""}${score.toFixed(4)} (${score > 0 ? "pushes toward hate" : "pushes toward non-hate"})`,
        style: cls === "neutral" ? "" : `opacity:${(0.45 + 0.55 * magnitude).toFixed(2)}`,
      });
      return node;
    })
  );
}

function renderResult(result, input) {
  const body = clear($("#results-body"));
  const verdict = threeWayVerdict(result);
  const exp = result.explanation;

  body.append(
    el("div", { class: "result-head" }, [
      el("span", { class: `verdict ${verdict}`, text: VERDICT_LABEL[verdict] }),
      el("span", { class: "muted small", text: `Language estimate: ${result.language}${result.code_mixed ? " · mixed scripts" : ""}` }),
    ])
  );
  if (verdict === "non_hate") {
    body.append(el("p", { class: "muted small", text: "Posts classified as non-hate can still contain insults or profanity." }));
  }

  // Display the score used by the moderation policy.
  const bars = el("div", { class: "prob-bars" });
  const fill = el("div", { class: "prob-fill hate" });
  bars.append(el("div", { class: "prob-row" }, [
    el("span", { class: "prob-label", text: "Hate score" }),
    el("div", { class: "prob-track" }, [fill]),
    el("span", { class: "prob-value", text: pct(result.probability_hate) }),
  ]));
  requestAnimationFrame(() => (fill.style.width = pct(result.probability_hate)));
  if (state.operating) {
    const { low, high } = state.operating;
    bars.append(el("p", { class: "muted small",
      text: `Scores at or above ${pct(high)} flag Hate. Scores at or below ${pct(low)} show Non-hate. Scores in between go to review.` }));
  }
  body.append(bars);

  // Uncertainty + faithfulness tiles
  const unc = result.uncertainty || {};
  const needsReview = verdict === "review";
  body.append(
    el("div", { class: `uncertainty ${needsReview || unc.is_uncertain ? "warn" : "ok"}` }, [
      el("strong", { text: unc.is_uncertain ? "Elevated model uncertainty" : "Model uncertainty" }),
      el("span", {
        class: "small muted",
        text: `margin ${num(unc.margin, 3)} · entropy ${num(unc.normalized_entropy, 3)}${
          needsReview ? " · routed to human review" : ""
        }`,
      }),
    ])
  );

  const faith = exp.faithfulness || {};
  body.append(
    el("div", { class: "grid grid-2", style: "margin-top:16px" }, [
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

  // Keep the moderation verdict and token evidence visible in the same result.
  body.append(el("h3", { style: "margin-top:20px", text: "Model-derived evidence" }));
  body.append(
    el("div", { class: "chips" }, (exp.top_evidence || []).map((t) => el("span", { class: "chip", text: t })))
  );
  body.append(el("div", { class: "disclaimer", style: "margin-top:12px",
    text: "The highlights explain which tokens changed the score. A human reviewer can assess intent and context." }));

  showAlert($("#form-alert"), "ok", "Analysis complete.");
  $("#report-btn").classList.remove("hidden");
  state.last = { result, input };
}

function setAnalyzing(active) {
  const button = clear($("#analyze-btn"));
  button.disabled = active;
  if (active) button.append(el("span", { class: "spinner" }), " Analyzing…");
  else button.textContent = "Analyze";
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

  const input = { text, context, method };
  state.pending = input;
  state.last = null;
  $("#report-btn").classList.add("hidden");
  setAnalyzing(true);
  clear(alertBox);
  renderSkeleton();

  try {
    const result = await API.post("/api/explain", input);
    if (state.pending === input) renderResult(result, input);
  } catch (err) {
    if (state.pending !== input) return;
    clear($("#results-body"));
    $("#results-body").append(el("div", { class: "empty", text: "Analysis failed." }));
    showAlert(alertBox, "error", err.message);
    $("#report-btn").classList.add("hidden");
  } finally {
    if (state.pending === input) {
      state.pending = null;
      setAnalyzing(false);
    }
  }
}

async function validateImage(event) {
  event.preventDefault();
  const file = $("#image-file").files[0];
  const alert = clear($("#image-alert"));
  const evidence = clear($("#meme-evidence"));
  if (!file) return;
  const body = new FormData();
  body.append("image", file);
  if (state.memeEnabled) body.append("caption", $("#meme-caption").value);
  const button = $("#image-submit");
  button.disabled = true;
  try {
    if (state.memeEnabled) {
      const result = await API.post("/api/meme-predict", body);
      showAlert(alert, "ok", `Pilot label: ${result.decision === "hate" ? "Hate" : "Non-hate"} · Hate score ${pct(result.probability_hate)}`);
      const url = URL.createObjectURL(file);
      const image = el("img", { src: url, alt: "Uploaded meme with a 3 by 3 attribution overlay" });
      image.addEventListener("load", () => URL.revokeObjectURL(url), { once: true });
      image.addEventListener("error", () => URL.revokeObjectURL(url), { once: true });
      evidence.append(
        el("div", { class: "meme-preview" }, [
          image,
          el("div", { class: "meme-regions" }, result.region_deltas.map((delta, index) =>
            el("span", { title: `Region ${index + 1}. Hate score change when covered: ${num(-delta)}`,
              style: `background:rgba(${delta >= 0 ? "255,70,100" : "65,155,255"},${Math.min(0.7, Math.abs(delta) * 5)})` }))),
        ]),
        el("p", { class: "muted small", text: "Covering red regions lowers the hate score. Covering blue regions raises it." }),
      );
    } else {
      const result = await API.post("/api/vision-check", body);
      showAlert(alert, "ok", `${result.format} · ${result.width} × ${result.height}. ${result.message}`);
    }
  } catch (err) {
    showAlert(alert, "error", err.message);
  } finally {
    button.disabled = false;
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  const textarea = $("#post-text");
  const counter = $("#char-counter");
  const showCounter = (len) => {
    counter.textContent = `${len} / 5000`;
    counter.classList.toggle("over", len > 4800);
  };
  const updateCounter = () => showCounter(textarea.value.length);
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
  $("#analyze-form").addEventListener("reset", () => {
    state.pending = null;
    state.last = null;
    setAnalyzing(false);
    clear($("#results-body"));
    $("#results-body").append(el("div", { class: "empty", text: "No analysis yet. Submit a post to see the decision and explanation." }));
    clear($("#form-alert"));
    $("#report-btn").classList.add("hidden");
    showCounter(textarea.defaultValue.length);
  });
  $("#image-form").addEventListener("submit", validateImage);

  $("#report-btn").addEventListener("click", () => {
    if (!state.last) return;
    const { result, input } = state.last;
    const verdict = threeWayVerdict(result);
    sessionStorage.setItem(
      "hatexai.feedback",
      JSON.stringify({
        text: input.text,
        context: input.context,
        predicted_label: verdict === "review" ? "needs_review" : verdict,
        probability_hate: result.probability_hate,
        model_version: result.model?.version || null,
      })
    );
    window.location.href = "feedback.html";
  });

  // Operating point drives the three-way verdict; if it is missing, we fall
  // back to the raw label rather than inventing a threshold.
  try {
    const [evaluation, info] = await Promise.all([API.get("/api/evaluation"), API.get("/api/model-info")]);
    state.operating = (evaluation.available && evaluation.available.threshold) || null;
    $("#analysis-scope").textContent = info.supported_languages.length
      ? `Held-out text evaluation: ${info.supported_languages.join(" and ")}.`
      : "The demo profile uses a small synthetic dataset.";
    if (info.modalities.image) {
      state.memeEnabled = true;
      $("#image-heading").textContent = "Analyze an English meme (research pilot)";
      $("#image-help").textContent = "Upload a PNG, JPEG, or WebP up to 5 MB and enter its caption. Scores are experimental.";
      $("#meme-caption-field").classList.remove("hidden");
      $("#meme-caption").required = true;
      $("#image-submit").textContent = "Analyze meme";
    }
  } catch {
    state.operating = null;
  }
});
