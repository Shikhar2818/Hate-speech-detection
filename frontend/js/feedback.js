/* HateXAI — feedback submission and review queue. */

const STATUS_CLASS = {
  pending_review: "warn",
  accepted: "ok",
  rejected: "bad",
  needs_discussion: "neutral",
};

const attached = { predicted_label: null, probability_hate: null, model_version: null };

function clearAttachment() {
  attached.predicted_label = null;
  attached.probability_hate = null;
  attached.model_version = null;
  $("#fb-prediction").textContent = "no analysis attached";
}

function renderQueue(records, summary) {
  const queue = clear($("#queue"));
  $("#queue-summary").textContent = summary ? `${summary.total} report(s) · ${Object.entries(summary.by_status).map(([k, v]) => `${v} ${k.replace("_", " ")}`).join(", ")}` : "";
  if (!records.length) {
    queue.append(el("div", { class: "empty", text: "No reports yet. Submitted feedback will appear here as review candidates." }));
    return;
  }
  records.slice(0, 20).forEach((r) => {
    const card = el("div", { class: "card", style: "padding:16px; background: var(--panel-2); margin-bottom:12px;" });
    const label = el("select", { "aria-label": "Reviewed label" }, [
      el("option", { value: "", text: "Choose a reviewed label (optional)" }),
      el("option", { value: "hate", text: "Hate" }),
      el("option", { value: "non_hate", text: "Non-hate" }),
    ]);
    label.value = r.reviewed_label || "";
    card.append(
      el("div", { class: "field-row" }, [
        el("span", { class: `pill ${STATUS_CLASS[r.status] || "neutral"}`, text: r.status.replace("_", " ") }),
        el("span", { class: "muted small mono", text: `${r.id} · ${(r.created_at || "").slice(0, 19).replace("T", " ")}` }),
      ]),
      el("p", { style: "margin:10px 0 6px", text: r.text.slice(0, 260) }),
      el("p", { class: "muted small", text: `Model said: ${r.predicted_label || "n/a"} (P(hate)=${r.probability_hate === null ? "n/a" : num(r.probability_hate, 3)}) · user says: ${r.user_label || "unspecified"}` }),
      r.comment ? el("p", { class: "muted small", text: `“${r.comment}”` }) : null,
      label,
      el("div", { class: "chips", style: "margin-top:10px" }, [
        el("button", { class: "chip", text: "Accept", onclick: () => review(r.id, "accepted", label.value || null) }),
        el("button", { class: "chip", text: "Reject", onclick: () => review(r.id, "rejected") }),
        el("button", { class: "chip", text: "Needs discussion", onclick: () => review(r.id, "needs_discussion") }),
      ])
    );
    queue.append(card);
  });
}

async function refreshQueue() {
  try {
    const [records, summary] = await Promise.all([API.get("/api/feedback?limit=50"), API.get("/api/feedback/summary")]);
    renderQueue(records, summary);
  } catch (err) {
    clear($("#queue"));
    $("#queue").append(el("div", { class: "empty", text: `Could not load queue: ${err.message}` }));
  }
}

async function review(id, status, reviewed_label = null) {
  try {
    await API.post(`/api/feedback/${id}/review`, { status, reviewed_label });
    await refreshQueue();
  } catch (err) {
    showAlert($("#feedback-alert"), "error", `Review update failed: ${err.message}`);
  }
}

async function submit(event) {
  event.preventDefault();
  const text = $("#fb-text").value;
  if (!text.trim()) {
    showAlert($("#feedback-alert"), "error", "Please include the post text.");
    return;
  }
  const userLabel = ($("input[name='user_label']:checked") || {}).value || null;
  const button = $("#fb-submit");
  button.disabled = true;
  try {
    const result = await API.post("/api/feedback", {
      text,
      context: $("#fb-context").value,
      predicted_label: attached.predicted_label,
      probability_hate: attached.probability_hate,
      model_version: attached.model_version,
      user_label: userLabel,
      comment: $("#fb-comment").value,
    });
    showAlert($("#feedback-alert"), "ok", `${result.message} (id ${result.id})`);
    $("#feedback-form").reset();
    clearAttachment();
    await refreshQueue();
  } catch (err) {
    showAlert($("#feedback-alert"), "error", err.message);
  } finally {
    button.disabled = false;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const stored = sessionStorage.getItem("hatexai.feedback");
  if (stored) {
    sessionStorage.removeItem("hatexai.feedback");
    try {
      const data = JSON.parse(stored);
      $("#fb-text").value = data.text || "";
      $("#fb-context").value = data.context || "";
      attached.predicted_label = data.predicted_label || null;
      attached.probability_hate = typeof data.probability_hate === "number" ? data.probability_hate : null;
      attached.model_version = data.model_version || null;
      $("#fb-prediction").textContent = attached.predicted_label
        ? `${attached.predicted_label} (P(hate)=${num(attached.probability_hate, 3)})`
        : "no analysis attached";
    } catch {
      /* ignore malformed prefill */
    }
  }
  $("#fb-text").addEventListener("input", clearAttachment);
  $("#fb-context").addEventListener("input", clearAttachment);
  $("#feedback-form").addEventListener("submit", submit);
  refreshQueue();
});
