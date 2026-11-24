/* HateXAI — research & model information. */

function list(items) {
  return el("ul", { class: "muted small", style: "padding-left:18px" }, items.map((t) => el("li", { text: t })));
}

function render(info) {
  $("#model-version").textContent = info.version;
  clear($("#model-arch")).append(
    el("p", {}, [el("strong", { text: "Served model: " }), document.createTextNode(info.architecture.served_model)]),
    el("p", { class: "muted small", text: "Optional components (implemented, availability reported honestly):" }),
    list(info.architecture.optional_models)
  );

  clear($("#model-modalities")).append(
    el("p", {}, [el("strong", { text: "Languages: " }), document.createTextNode((info.supported_languages || []).join(", "))]),
    el("div", { class: "chips" }, [
      el("span", { class: `pill ${info.modalities.text ? "ok" : "neutral"}`, text: "text" }),
      el("span", { class: `pill ${info.modalities.context ? "ok" : "neutral"}`, text: "context" }),
      el("span", { class: `pill ${info.modalities.image ? "ok" : "bad"}`, text: "image" }),
    ]),
    el("p", { class: "muted small", style: "margin-top:10px", text: info.modalities.image_note })
  );

  const d = info.dataset;
  clear($("#model-datasets")).append(
    el("p", {}, [el("strong", { text: "Shipped: " }), document.createTextNode(d.shipped)]),
    el("p", {}, [el("strong", { text: "Probes: " }), document.createTextNode(d.probes)]),
    el("p", { class: "muted small", text: "Recommended public corpora (convert with scripts/prepare_public_data.py):" }),
    list(d.recommended_public)
  );

  const limitations = clear($("#model-limitations"));
  (info.known_limitations || []).forEach((t) => limitations.append(el("li", { text: t })));
  clear($("#model-responsible")).append(el("p", { class: "small", text: info.responsible_use }));
}

document.addEventListener("DOMContentLoaded", async () => {
  try {
    const info = await API.get("/api/model-info");
    render(info);
  } catch (err) {
    showAlert($("#research-alert"), "error", err.message);
  }
});
