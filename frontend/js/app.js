/* HateXAI — shared frontend utilities.
   Vanilla JS only: no framework, no build step. */

const API = {
  base: "",
  async get(path) {
    return request("GET", path);
  },
  async post(path, body) {
    return request("POST", path, body);
  },
};

async function request(method, path, body) {
  const options = { method, headers: {} };
  if (body instanceof FormData) {
    options.body = body;
  } else if (body !== undefined) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  let response;
  try {
    response = await fetch(API.base + path, options);
  } catch (err) {
    throw new ApiError("Cannot reach the backend. Is it running on this origin?", 0, null);
  }
  const text = await response.text();
  let payload = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = { detail: text.slice(0, 300) };
    }
  }
  if (!response.ok) {
    throw new ApiError(extractDetail(payload, response.status), response.status, payload);
  }
  return payload;
}

class ApiError extends Error {
  constructor(message, status, payload) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.payload = payload;
  }
}

function extractDetail(payload, status) {
  if (!payload) return `Request failed (HTTP ${status})`;
  const detail = payload.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail.length) {
    // FastAPI validation errors
    const first = detail[0];
    const loc = (first.loc || []).filter((x) => x !== "body").join(".");
    return `${loc ? loc + ": " : ""}${first.msg || "invalid input"}`;
  }
  return `Request failed (HTTP ${status})`;
}

/* ---------- DOM helpers ---------- */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "html") node.innerHTML = value;
    else if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2), value);
    else if (value !== null && value !== undefined && value !== false) node.setAttribute(key, value);
  }
  for (const child of [].concat(children)) {
    if (child == null) continue;
    node.append(typeof child === "string" ? document.createTextNode(child) : child);
  }
  return node;
}

function clear(node) {
  while (node && node.firstChild) node.removeChild(node.firstChild);
  return node;
}

function pct(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  return `${(value * 100).toFixed(digits)}%`;
}

function num(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  return Number(value).toFixed(digits);
}

function showAlert(container, kind, message) {
  if (!container) return;
  clear(container);
  container.append(el("div", { class: `alert ${kind}`, text: message }));
}

/* ---------- Page chrome ---------- */

function initChrome() {
  const page = document.body.dataset.page;
  $$(".nav a").forEach((a) => {
    if (a.dataset.nav === page) a.classList.add("active");
  });
  const year = $("#year");
  if (year) year.textContent = new Date().getFullYear();
}

function initReveal() {
  const nodes = $$(".reveal");
  if (!("IntersectionObserver" in window) || window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    nodes.forEach((n) => n.classList.add("in"));
    return;
  }
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("in");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.12 }
  );
  nodes.forEach((n) => observer.observe(n));
}

document.addEventListener("DOMContentLoaded", () => {
  initChrome();
  initReveal();
});
