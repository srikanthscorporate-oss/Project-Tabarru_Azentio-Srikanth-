// Adjuster view. Every value shown comes from the API; nothing is invented here.
// All API-derived text is inserted with textContent (via el()), never innerHTML.
"use strict";

const DECISION_LABEL = {
  COVERED: "Covered",
  PARTIALLY_COVERED: "Partially covered",
  NOT_COVERED: "Not covered",
  REFER: "Needs review",
};
const STATUS_LABEL = {
  pending_adjuster: "Pending adjuster",
  decided: "Decided by adjuster",
  intake_complete: "Intake complete",
  flagged: "Flagged",
  failed: "Failed",
};
const CODE_LABEL = {
  CERT_NOT_FOUND: "Certificate not found or not active on the incident date",
  MISSING_DOCUMENT: "Required document missing",
  DATE_ORDER_INVALID: "Dates are in an impossible order",
  AMOUNT_MISMATCH: "Claimed amount differs from the bill or quotation by more than 1%",
  FIELDS_INCOMPLETE: "Required fields could not be extracted",
  CITATION_INVALID: "Model citations failed the check after a retry",
  COVERAGE_OUTPUT_INVALID: "Model coverage output was invalid after a retry",
  PROVIDER_ERROR: "The model provider returned an error",
  CALC_INPUT_MISSING: "Inputs for the payable calculation are missing",
  INTAKE_FAILED: "No document could be processed",
  INTAKE_INCOMPLETE: "Intake did not complete cleanly",
};
const FIELD_LABEL = [
  ["certificate_no", "Certificate number"],
  ["product", "Product"],
  ["participant_name", "Participant"],
  ["participant_ic", "Participant IC (masked)"],
  ["incident_date", "Incident date"],
  ["admission_date", "Admission date"],
  ["discharge_date", "Discharge date"],
  ["diagnosis", "Diagnosis"],
  ["hospital_name", "Hospital"],
  ["room_rate_per_day", "Room rate per day", "money"],
  ["bill_total", "Bill total", "money"],
  ["vehicle_reg_no", "Vehicle registration"],
  ["cause_of_loss", "Cause of loss"],
  ["repair_total", "Repair quotation total", "money"],
  ["police_report_no", "Police report number"],
];
const MEDICAL_ONLY = new Set(["admission_date", "discharge_date", "diagnosis", "hospital_name", "room_rate_per_day", "bill_total"]);
const MOTOR_ONLY = new Set(["vehicle_reg_no", "cause_of_loss", "repair_total", "police_report_no"]);
const ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/;
const MONEY_PATTERN = /^\d{1,12}(\.\d{1,2})?$/;

const state = { queue: [], queueError: null, loading: false, selected: null, claim: null, recent: new Map(), submitting: false };

// ---- DOM helper: children are nodes or strings (strings become text nodes)
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}
const $ = (id) => document.getElementById(id);

// MYR from the backend's exact decimal string: grouping only, no float conversion.
function myr(value) {
  if (value === null || value === undefined) return el("span", { class: "missing" }, "Not available");
  const [whole, frac = "00"] = String(value).split(".");
  const grouped = whole.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  return el("span", { class: "num" }, `MYR ${grouped}.${frac.padEnd(2, "0")}`);
}
function show(value) {
  return value === null || value === undefined || value === "" ? el("span", { class: "missing" }, "Not available") : String(value);
}
function badge(decision) {
  return el("span", { class: `badge b-${decision}` }, DECISION_LABEL[decision] || decision);
}
function codeChip(code) {
  return el("li", { class: "code-chip" }, el("abbr", { title: code }, CODE_LABEL[code] || code), " ", el("span", { class: "code" }, code));
}

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  let body = null;
  try { body = await response.json(); } catch { body = null; }
  return { status: response.status, ok: response.ok, body };
}

// ---- Queue
async function loadQueue() {
  state.loading = true;
  state.queueError = null;
  renderQueue();
  try {
    const res = await api("/v1/adjuster/queue");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    state.queue = res.body;
  } catch (err) {
    state.queueError = "The queue could not be loaded. Check that the API is running, then refresh.";
  } finally {
    state.loading = false;
    renderQueue();
  }
}

function filteredQueue() {
  const term = $("search").value.trim().toLowerCase();
  const product = $("product").value;
  const decision = $("decision").value;
  return state.queue.filter((item) =>
    (!term || item.claim_id.toLowerCase().includes(term)) &&
    (!product || item.product === product) &&
    (!decision || item.recommendation.decision === decision));
}

function queueButton(item) {
  const rec = item.recommendation;
  const codes = rec.reason_codes.length ? rec.reason_codes.join(", ") : "No reason codes";
  return el("li", {},
    el("button", {
      type: "button", class: "queue-item",
      "aria-current": state.selected === item.claim_id ? "true" : "false",
      onclick: () => openClaim(item.claim_id),
    },
      el("span", { class: "qi-top" }, el("span", { class: "qi-id" }, item.claim_id), badge(rec.decision)),
      el("span", { class: "qi-bottom" }, el("span", {}, item.product ? item.product.toLowerCase().replace(/^./, (c) => c.toUpperCase()) : "Product unknown"), myr(rec.payable_amount)),
      el("span", { class: "qi-codes" }, codes)));
}

function renderQueue() {
  const list = $("queue-list");
  const summary = $("queue-summary");
  list.replaceChildren();
  if (state.loading) { summary.textContent = "Loading the queue..."; return; }
  if (state.queueError) {
    summary.textContent = "";
    list.append(el("li", { class: "state-msg error", role: "alert" }, state.queueError));
    return;
  }
  const items = filteredQueue();
  const refer = state.queue.filter((i) => i.recommendation.decision === "REFER").length;
  summary.textContent = `${state.queue.length} pending in the loaded queue, ${refer} needing review. Showing ${items.length}.`;
  if (!state.queue.length) list.append(el("li", { class: "state-msg" }, "No claims are waiting for a decision."));
  else if (!items.length) list.append(el("li", { class: "state-msg" }, "No pending claims match these filters."));
  else list.append(...items.map(queueButton));

  const recent = $("recent-list");
  recent.replaceChildren();
  for (const claim of state.recent.values()) {
    const decided = claim.adjuster_decision;
    recent.append(el("li", {}, el("button", {
      type: "button", class: "queue-item", "aria-current": state.selected === claim.claim_id ? "true" : "false",
      onclick: () => openClaim(claim.claim_id),
    },
      el("span", { class: "qi-top" }, el("span", { class: "qi-id" }, claim.claim_id),
        el("span", { class: "badge b-neutral" }, STATUS_LABEL[claim.status] || claim.status)),
      decided ? el("span", { class: "qi-bottom" }, el("span", {}, `Final: ${DECISION_LABEL[decided.final_decision]}`), myr(decided.final_payable)) : null)));
  }
}

// ---- Claim
async function openClaim(claimId, notice = null) {
  state.selected = claimId;
  renderQueue();
  const review = $("review");
  review.setAttribute("aria-busy", "true");
  review.replaceChildren(el("p", { class: "state-msg" }, `Loading ${claimId}...`));
  $("decision-panel").replaceChildren();
  let res;
  try {
    res = await api(`/v1/claims/${encodeURIComponent(claimId)}`);
  } catch {
    review.replaceChildren(el("p", { class: "state-msg error", role: "alert" }, "The claim could not be loaded. Check the connection and try again."));
    review.setAttribute("aria-busy", "false");
    return null;
  }
  review.setAttribute("aria-busy", "false");
  if (res.status === 404) {
    review.replaceChildren(el("p", { class: "state-msg error", role: "alert" }, "No claim with that ID exists."));
    return null;
  }
  if (!res.ok) {
    review.replaceChildren(el("p", { class: "state-msg error", role: "alert" }, `The claim could not be loaded (HTTP ${res.status}).`));
    return null;
  }
  state.claim = res.body;
  state.recent.set(claimId, res.body);
  renderQueue();
  renderClaim(res.body);
  renderDecision(res.body, notice);
  loadAudit(claimId);
  return res.body;
}

function section(title, ...content) {
  return el("section", { "aria-label": title }, el("h2", {}, title), ...content);
}

function renderClaim(claim) {
  const rec = claim.recommendation;
  const f = claim.fields;
  const decided = claim.adjuster_decision;
  const product = f.product;

  const head = el("div", { class: "claim-head" },
    el("h2", { class: "claim-title" }, claim.claim_id),
    el("dl", { class: "statuses" },
      el("div", {}, el("dt", {}, "Workflow status"), el("dd", {}, el("span", { class: "badge b-neutral" }, STATUS_LABEL[claim.status] || claim.status))),
      el("div", {}, el("dt", {}, "Intake status"), el("dd", {}, el("span", { class: "badge b-neutral" }, STATUS_LABEL[claim.intake_status] || show(claim.intake_status)))),
      rec ? el("div", {}, el("dt", {}, "Recommendation"), el("dd", {}, badge(rec.decision))) : null,
      decided ? el("div", {}, el("dt", {}, "Final decision"), el("dd", {}, badge(decided.final_decision))) : null));

  const amounts = el("dl", { class: "amounts" },
    el("div", { class: "amount" }, el("dt", {}, "Claimed"), el("dd", {}, myr(f.amount_claimed))),
    el("div", { class: "amount" }, el("dt", {}, "Recommended payable"), el("dd", {}, rec ? myr(rec.payable_amount) : show(null))),
    el("div", { class: "amount final" }, el("dt", {}, "Final payable (adjuster)"),
      el("dd", {}, decided ? myr(decided.final_payable) : el("span", { class: "missing" }, "Not decided yet"))));

  const facts = el("dl", { class: "facts" },
    FIELD_LABEL.filter(([key]) => !(product === "MEDICAL" && MOTOR_ONLY.has(key)) && !(product === "MOTOR" && MEDICAL_ONLY.has(key)))
      .map(([key, label, kind]) => el("div", {}, el("dt", {}, label), el("dd", {}, kind === "money" ? myr(f[key]) : show(f[key])))));

  const docs = el("div", { class: "table-wrap" }, el("table", {},
    el("caption", { class: "hint", style: "text-align:left" }, "Document IDs are the submitted file names; the type comes from classification."),
    el("thead", {}, el("tr", {}, el("th", { scope: "col" }, "Document"), el("th", { scope: "col" }, "Classified type"), el("th", { scope: "col" }, "Status"))),
    el("tbody", {}, claim.documents.map((d) => el("tr", {},
      el("td", { class: "code" }, d.doc_id), el("td", {}, show(d.doc_type)),
      el("td", {}, d.status === "failed" ? el("strong", {}, "Failed") : "Processed"))))));

  const intakeCodes = claim.reason_codes.length
    ? el("ul", { class: "codes" }, claim.reason_codes.map(codeChip))
    : el("p", { class: "hint" }, "No intake reason codes.");
  const recCodes = rec && rec.reason_codes.filter((c) => !claim.reason_codes.includes(c));

  const recommendation = rec ? section("Recommendation",
    el("p", {}, badge(rec.decision), " ", rec.decision === "REFER" ? "This claim needs an adjuster's review; it is not a rejection." : ""),
    el("p", { class: "rationale" }, rec.rationale || "No rationale was returned."),
    recCodes.length ? el("ul", { class: "codes", "aria-label": "Coverage reason codes" }, recCodes.map(codeChip)) : null,
    citations(rec, product)) : section("Recommendation", el("p", { class: "missing" }, "Not available"));

  $("review").replaceChildren(
    head,
    section("Amounts", amounts),
    section("Claim facts", facts),
    section("Documents", docs),
    section("Intake reason codes", intakeCodes),
    recommendation,
    section("Processing trace", trace(claim)));
}

function citations(rec, product) {
  if (!rec.cited_clauses.length) return el("p", { class: "hint" }, "No clauses were cited.");
  return el("div", {},
    el("h3", { style: "margin-top:16px" }, `Cited clauses (${rec.cited_clauses.length})`),
    el("p", { class: "hint" }, "Each stored quote passed the backend check that it is an exact substring of a clause retrieved for this claim."),
    rec.cited_clauses.map((c) => {
      const sourceBox = el("div", { class: "source", hidden: true });
      const toggle = el("button", { type: "button", class: "btn btn-quiet", "aria-expanded": "false" }, "Show source clause");
      toggle.addEventListener("click", () => toggleSource(toggle, sourceBox, product, c));
      return el("article", { class: "citation" },
        el("div", { class: "citation-head" },
          el("h3", {}, el("span", { class: "code" }, c.clause_id), " ", el("span", { class: "hint" }, product ? `${product.toLowerCase()} certificate` : "")),
          product ? toggle : null),
        el("blockquote", { class: "quote" }, c.quote),
        sourceBox);
    }));
}

async function toggleSource(button, box, product, cited) {
  const open = button.getAttribute("aria-expanded") === "true";
  if (open) { box.hidden = true; button.setAttribute("aria-expanded", "false"); button.textContent = "Show source clause"; return; }
  button.setAttribute("aria-expanded", "true");
  button.textContent = "Hide source clause";
  box.hidden = false;
  if (box.dataset.loaded) return;
  box.replaceChildren("Loading the clause...");
  try {
    const res = await api(`/v1/certificates/${encodeURIComponent(product)}/clauses/${encodeURIComponent(cited.clause_id)}`);
    if (!res.ok) { box.replaceChildren(res.status === 404 ? "This clause is not in the current certificate wording." : `Could not load the clause (HTTP ${res.status}).`); return; }
    const text = res.body.text;
    const at = text.indexOf(cited.quote);
    const body = at < 0
      ? [text]
      : [text.slice(0, at), el("mark", {}, text.slice(at, at + cited.quote.length)), text.slice(at + cited.quote.length)];
    box.replaceChildren(
      el("p", { class: "hint" }, "Current certificate wording, not a historical snapshot. The highlight shows where the quote appears."),
      el("p", { style: "margin-top:8px" }, el("span", { class: "code" }, res.body.clause_id), " ", ...body),
      at < 0 ? el("p", { class: "hint" }, "The quote does not appear in the current wording, which may have changed since assessment.") : null);
    box.dataset.loaded = "1";
  } catch {
    box.replaceChildren("Could not load the clause. Check the connection.");
  }
}

function trace(claim) {
  const calls = claim.trace.reduce((n, t) => n + t.provider_calls, 0);
  const coverage = claim.trace.find((t) => t.step === "coverage");
  const skippedNote = coverage && coverage.status === "skipped"
    ? el("p", { class: "form-msg info" }, "Coverage was not assessed: intake reported problems (",
      claim.reason_codes.map((c) => CODE_LABEL[c] || c).join("; ") || "intake did not complete",
      "), so the claim was referred for review without a model call.")
    : null;
  return el("div", {}, skippedNote,
    el("details", { class: "trace" },
      el("summary", {}, `${claim.trace.length} steps, ${calls} provider calls`),
      el("ol", { class: "steps" }, claim.trace.map((t) => el("li", { class: "step" },
        el("span", { class: `step-dot ${t.status}`, "aria-hidden": "true" }),
        el("div", {},
          el("p", {}, el("strong", {}, t.step), ` : ${t.status}`),
          el("p", { class: "step-meta num" }, `${t.duration_ms.toFixed(2)} ms, ${t.provider_calls} provider call${t.provider_calls === 1 ? "" : "s"}, documents: ${t.doc_ids.join(", ") || "none"}`),
          t.clause_ids && t.clause_ids.length ? el("p", { class: "step-meta" }, `Retrieved clauses: ${t.clause_ids.join(", ")}`) : null,
          t.error ? el("p", { class: "step-error" }, `Error: ${t.error}`) : null))))));
}

// ---- Audit
async function loadAudit(claimId) {
  const holder = $("audit-list");
  if (!holder) return;
  try {
    const res = await api(`/v1/claims/${encodeURIComponent(claimId)}/audit`);
    if (state.selected !== claimId) return;
    if (!res.ok) { holder.replaceChildren(el("li", {}, `Audit history unavailable (HTTP ${res.status}).`)); return; }
    holder.replaceChildren(...(res.body.length ? res.body.map((e) => el("li", {},
      el("strong", {}, e.event.replace(/_/g, " ")),
      el("span", { class: "num" }, e.timestamp),
      el("span", {}, `By ${e.actor}`),
      el("span", {}, `${DECISION_LABEL[e.decision] || e.decision}, `, myr(e.payable)),
      e.clause_ids.length ? el("span", {}, `Clauses: ${e.clause_ids.join(", ")}`) : null,
      e.reason ? el("span", {}, `Reason: ${e.reason}`) : null)) : [el("li", {}, "No audit events.")]));
  } catch {
    holder.replaceChildren(el("li", {}, "Audit history could not be loaded."));
  }
}

// ---- Decision
function renderDecision(claim, notice) {
  const panel = $("decision-panel");
  const parts = [el("h2", {}, "Decision")];
  if (notice) parts.push(el("p", { class: `form-msg ${notice.kind}`, role: notice.kind === "error" ? "alert" : "status" }, notice.text));
  const rec = claim.recommendation;
  if (claim.adjuster_decision) {
    const d = claim.adjuster_decision;
    parts.push(el("dl", { class: "confirm-summary" },
      el("dt", {}, "Final decision"), el("dd", {}, badge(d.final_decision)),
      el("dt", {}, "Final payable"), el("dd", {}, myr(d.final_payable)),
      el("dt", {}, "Action"), el("dd", {}, d.action === "approve" ? "Approved the recommendation" : "Overrode the recommendation"),
      el("dt", {}, "Adjuster"), el("dd", {}, d.adjuster),
      d.reason ? el("dt", {}, "Reason") : null, d.reason ? el("dd", {}, d.reason) : null,
      el("dt", {}, "Original recommendation"), el("dd", {}, rec ? badge(rec.decision) : show(null), " ", rec ? myr(rec.payable_amount) : "")));
  } else if (claim.status !== "pending_adjuster" || !rec) {
    parts.push(el("p", { class: "hint" }, "This claim is not waiting for an adjuster decision."));
  } else {
    parts.push(decisionForm(claim));
  }
  parts.push(el("h3", { style: "margin-top:24px" }, "Audit history"),
    el("p", { class: "hint" }, "Read-only. The database blocks edits and deletes; this is not cryptographic tamper proofing."),
    el("ul", { class: "audit", id: "audit-list" }, el("li", {}, "Loading...")));
  panel.replaceChildren(...parts);
}

function decisionForm(claim) {
  const rec = claim.recommendation;
  const msg = el("div", { id: "form-msg", "aria-live": "assertive" });
  const errorFor = (id) => el("p", { class: "field-error", id: `${id}-error`, hidden: true });
  const adjuster = el("input", { id: "adjuster", maxlength: "64", autocomplete: "off", "aria-describedby": "adjuster-hint adjuster-error" });
  const approve = el("input", { type: "radio", name: "action", value: "approve", checked: true });
  const override = el("input", { type: "radio", name: "action", value: "override" });
  const finalDecision = el("select", { id: "final-decision", "aria-describedby": "final-decision-error" },
    Object.entries(DECISION_LABEL).map(([v, l]) => el("option", { value: v }, l)));
  finalDecision.value = rec.decision;
  const payable = el("input", { id: "final-payable", inputmode: "decimal", value: rec.payable_amount, "aria-describedby": "final-payable-hint final-payable-error" });
  const reason = el("textarea", { id: "reason", maxlength: "2000", "aria-describedby": "reason-error" });
  const overrideFields = el("div", { hidden: true },
    el("div", { class: "field" }, el("label", { for: "final-decision" }, "Final decision"), finalDecision, errorFor("final-decision")),
    el("div", { class: "field" }, el("label", { for: "final-payable" }, "Final payable (MYR)"), payable,
      el("p", { class: "hint", id: "final-payable-hint" }, "Two decimal places. 0.00 for Not covered and Needs review."), errorFor("final-payable")),
    el("div", { class: "field" }, el("label", { for: "reason" }, "Override reason (required)"), reason, errorFor("reason")));
  const submit = el("button", { type: "submit", class: "btn btn-primary btn-block" }, "Review and record");

  const sync = () => {
    overrideFields.hidden = !override.checked;
    submit.textContent = override.checked ? "Review override" : "Review approval";
  };
  approve.addEventListener("change", sync);
  override.addEventListener("change", sync);
  finalDecision.addEventListener("change", () => {
    if (finalDecision.value === "NOT_COVERED" || finalDecision.value === "REFER") payable.value = "0.00";
  });
  sync();

  const form = el("form", { novalidate: true },
    msg,
    el("div", { class: "field" }, el("label", { for: "adjuster" }, "Adjuster name or ID"), adjuster,
      el("p", { class: "hint", id: "adjuster-hint" }, "Recorded in the audit log. This is not a sign-in."), errorFor("adjuster")),
    el("fieldset", { class: "choice" }, el("legend", {}, "Action"),
      el("label", { class: "radio" }, approve, el("span", {}, "Approve the recommendation: ", badge(rec.decision), " ", myr(rec.payable_amount))),
      el("label", { class: "radio" }, override, el("span", {}, "Override with a different decision or amount"))),
    overrideFields,
    submit);

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (state.submitting) return;
    const fieldErrors = {};
    const body = { adjuster: adjuster.value.trim(), action: override.checked ? "override" : "approve" };
    if (!body.adjuster) fieldErrors.adjuster = "Enter your name or ID.";
    if (override.checked) {
      body.final_decision = finalDecision.value;
      body.final_payable = payable.value.trim();
      body.reason = reason.value.trim();
      if (!body.reason) fieldErrors.reason = "An override needs a reason.";
      if (!MONEY_PATTERN.test(body.final_payable)) fieldErrors["final-payable"] = "Enter an amount such as 1250.00.";
      else if (["NOT_COVERED", "REFER"].includes(body.final_decision) && Number(body.final_payable) !== 0) fieldErrors["final-payable"] = "Must be 0.00 for this decision.";
      else if (body.final_decision === "COVERED" && Number(body.final_payable) <= 0) fieldErrors["final-payable"] = "Covered needs an amount above 0.00.";
      if (body.final_decision === rec.decision && body.final_payable && Number(body.final_payable) === Number(rec.payable_amount)) {
        fieldErrors["final-decision"] = "This matches the recommendation. Choose Approve instead.";
      }
    }
    showFieldErrors(form, fieldErrors);
    if (Object.keys(fieldErrors).length) {
      msg.replaceChildren(el("p", { class: "form-msg error", role: "alert" }, "Fix the highlighted fields."));
      return;
    }
    msg.replaceChildren();
    const finalDec = body.action === "approve" ? rec.decision : body.final_decision;
    const finalPay = body.action === "approve" ? rec.payable_amount : body.final_payable;
    if (!(await confirmDecision(claim.claim_id, body, finalDec, finalPay))) return;
    await submitDecision(claim.claim_id, body, form, submit, msg);
  });
  return form;
}

function showFieldErrors(form, errors) {
  for (const id of ["adjuster", "final-decision", "final-payable", "reason"]) {
    const input = form.querySelector(`#${id}`);
    const holder = form.querySelector(`#${id}-error`);
    if (!input || !holder) continue;
    if (errors[id]) { holder.textContent = errors[id]; holder.hidden = false; input.setAttribute("aria-invalid", "true"); }
    else { holder.textContent = ""; holder.hidden = true; input.removeAttribute("aria-invalid"); }
  }
  const first = Object.keys(errors)[0];
  if (first) form.querySelector(`#${first}`)?.focus();
}

function confirmDecision(claimId, body, decision, payable) {
  const dialog = $("confirm");
  $("confirm-summary").replaceChildren(
    el("dt", {}, "Claim"), el("dd", {}, claimId),
    el("dt", {}, "Action"), el("dd", {}, body.action === "approve" ? "Approve recommendation" : "Override"),
    el("dt", {}, "Final decision"), el("dd", {}, badge(decision)),
    el("dt", {}, "Final payable"), el("dd", {}, myr(payable)),
    el("dt", {}, "Adjuster"), el("dd", {}, body.adjuster),
    body.reason ? el("dt", {}, "Reason") : null, body.reason ? el("dd", {}, body.reason) : null);
  return new Promise((resolve) => {
    dialog.addEventListener("close", () => resolve(dialog.returnValue === "ok"), { once: true });
    dialog.returnValue = "";
    dialog.showModal();
    $("confirm-ok").focus();
  });
}

async function submitDecision(claimId, body, form, submit, msg) {
  state.submitting = true;
  submit.disabled = true;
  submit.textContent = "Recording...";
  let res = null;
  try {
    res = await api(`/v1/claims/${encodeURIComponent(claimId)}/adjuster-decision`, { method: "POST", body: JSON.stringify(body) });
  } catch {
    res = null;
  }
  state.submitting = false;
  if (res === null) {
    // The request may or may not have reached the server: read the current state before any retry.
    const current = await openClaim(claimId);
    await loadQueue();
    if (current && current.status === "decided") {
      renderDecision(current, { kind: "info", text: "The connection dropped, but the claim shows as decided. Check that the recorded decision below is yours." });
    } else if (current) {
      renderDecision(current, { kind: "error", text: "The connection dropped and no decision was recorded. You can submit again." });
    }
    return;
  }
  if (res.ok) {
    state.recent.set(claimId, res.body);
    await loadQueue();
    await openClaim(claimId, { kind: "ok", text: "Decision recorded. The claim has left the pending queue." });
    return;
  }
  if (res.status === 409) {
    await loadQueue();
    await openClaim(claimId, { kind: "info", text: "This claim is no longer pending: another decision was already recorded. The current state is shown." });
    return;
  }
  submit.disabled = false;
  submit.textContent = "Review and record";
  const detail = res.body && res.body.detail;
  let text = `The decision was not recorded (HTTP ${res.status}).`;
  if (typeof detail === "string") text = `The decision was not recorded: ${detail}.`;
  else if (Array.isArray(detail)) {
    const map = { adjuster: "adjuster", final_payable: "final-payable", final_decision: "final-decision", reason: "reason" };
    const errs = {};
    for (const d of detail) {
      const key = map[d.loc[d.loc.length - 1]];
      if (key) errs[key] = d.msg;
    }
    showFieldErrors(form, errs);
    text = "The server rejected some fields. See the messages below each field.";
  }
  msg.replaceChildren(el("p", { class: "form-msg error", role: "alert" }, text));
}

// ---- Wiring
document.addEventListener("DOMContentLoaded", () => {
  $("refresh").addEventListener("click", loadQueue);
  for (const id of ["search", "product", "decision"]) $(id).addEventListener("input", renderQueue);
  $("lookup").addEventListener("submit", (event) => {
    event.preventDefault();
    const id = $("lookup-id").value.trim();
    if (!ID_PATTERN.test(id)) { $("lookup-id").setAttribute("aria-invalid", "true"); return; }
    $("lookup-id").removeAttribute("aria-invalid");
    openClaim(id);
  });
  loadQueue();
});
