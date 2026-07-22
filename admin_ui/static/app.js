const state = {
  csrf: "",
  files: [],
  corpusPreview: null,
  jobs: [],
  documents: [],
  selectedJobId: null,
  activeView: "ingest",
  activeSource: "upload",
  ocrDefault: false,
  ocrDefaultInitialized: false,
  corpusOcrOverridden: false,
  pollTimer: null,
};

const $ = (id) => document.getElementById(id);

async function request(path, options = {}) {
  const method = (options.method || "GET").toUpperCase();
  const headers = new Headers(options.headers || {});
  if (["POST", "DELETE"].includes(method)) headers.set("X-CSRF-Token", state.csrf);
  if (options.json !== undefined) {
    headers.set("Content-Type", "application/json");
    options.body = JSON.stringify(options.json);
  }
  const response = await fetch(path, { ...options, method, headers, credentials: "same-origin" });
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) {
    const detail = typeof payload === "object" ? payload.detail : payload;
    const error = new Error(detail || `Request failed with ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return payload;
}

function showLogin() {
  stopPolling();
  state.csrf = "";
  state.files = [];
  state.corpusPreview = null;
  state.jobs = [];
  state.documents = [];
  state.selectedJobId = null;
  state.activeSource = "upload";
  state.ocrDefault = false;
  state.ocrDefaultInitialized = false;
  state.corpusOcrOverridden = false;
  $("loginView").hidden = false;
  $("appView").hidden = true;
  $("fileInput").value = "";
  $("folderInput").value = "";
  $("corpusSubtree").value = "";
  $("corpusOcr").checked = false;
  $("corpusPrune").checked = false;
  $("corpusPrune").disabled = false;
  $("queueCorpus").disabled = true;
  $("documentSearch").value = "";
  $("documentType").value = "";
  $("documentSubject").value = "";
  $("documentGrade").value = "";
  $("documentLanguage").value = "";
  selectSource("upload");
  renderStaging();
  renderCorpusPreview();
  renderJobs();
  renderDocuments();
  clearJobDetails();
  renderServiceStatus(null, null);
}

function showApp() {
  $("loginView").hidden = true;
  $("appView").hidden = false;
  selectView("ingest");
  selectSource("upload");
  startPolling();
}

async function bootstrap() {
  try {
    const session = await request("/api/session");
    state.csrf = session.csrf_token;
    showApp();
    await refreshAll();
  } catch {
    showLogin();
  }
}

let refreshing = false;

function selectView(name) {
  state.activeView = name;
  document.querySelectorAll("[data-view]").forEach((button) => {
    button.setAttribute("aria-selected", String(button.dataset.view === name));
  });
  document.querySelectorAll("[data-view-panel]").forEach((panel) => {
    panel.hidden = panel.dataset.viewPanel !== name;
  });
  if (name === "documents") loadDocuments().catch(showError);
  if (name === "jobs") loadJobs().catch(showError);
}

function selectSource(name) {
  state.activeSource = name;
  document.querySelectorAll("[data-source]").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.source === name));
  });
  $("uploadPanel").hidden = name !== "upload";
  $("corpusPanel").hidden = name !== "corpus";
}

function stopPolling() {
  if (state.pollTimer) window.clearTimeout(state.pollTimer);
  state.pollTimer = null;
}

function hasActiveJobs() {
  return state.jobs.some((job) => ["queued", "running"].includes(job.status));
}

function initializeOcrDefault(value) {
  if (state.ocrDefaultInitialized) return;
  state.ocrDefault = Boolean(value);
  state.ocrDefaultInitialized = true;
  for (const row of state.files) {
    if (!row.ocrOverridden) row.ocr = state.ocrDefault;
  }
  if (!state.corpusOcrOverridden) $("corpusOcr").checked = state.ocrDefault;
  renderStaging();
}

async function refreshAll() {
  if (refreshing || $("appView").hidden) return;
  refreshing = true;
  try {
    const status = await request("/api/admin/ingestion/status");
    initializeOcrDefault(status.ocr_default);
    const corpusStatus = await request("/api/admin/corpus_status");
    await loadJobs();
    await refreshSelectedJob();
    renderServiceStatus(status, corpusStatus);
    if (state.activeView === "documents") await loadDocuments(corpusStatus);
  } finally {
    refreshing = false;
  }
}

function startPolling() {
  stopPolling();
  const tick = async () => {
    try {
      await refreshAll();
    } catch (error) {
      showError(error);
    } finally {
      if (!$("appView").hidden) {
        state.pollTimer = window.setTimeout(tick, hasActiveJobs() ? 2000 : 10000);
      }
    }
  };
  state.pollTimer = window.setTimeout(tick, 0);
}

function statusBadge(label, stateName) {
  const badge = document.createElement("span");
  badge.className = `badge ${stateName}`;
  badge.textContent = label;
  return badge;
}

function renderServiceStatus(status, corpusStatus) {
  const strip = $("serviceStatus");
  if (!status || !corpusStatus) {
    strip.replaceChildren();
    return;
  }
  const qdrantReady = ["ready", "empty"].includes(corpusStatus.status);
  const active = state.jobs.find((job) => job.status === "running");
  const badges = [
    statusBadge("Backend online", "success"),
    statusBadge(`Qdrant ${corpusStatus.status}`, qdrantReady ? "success" : "warning"),
    statusBadge(status.worker.online ? "Worker online" : "Worker offline", status.worker.online ? "success" : "danger"),
    statusBadge(`Queued ${status.queue.queued || 0}`, status.queue.queued ? "warning" : ""),
    statusBadge(`Running ${status.queue.running || 0}`, status.queue.running ? "warning" : ""),
  ];
  if (active) {
    const activity = [active.current_item, active.current_stage].filter(Boolean).join(": ");
    badges.push(statusBadge(`Active ${activity || active.id.slice(0, 8)}`, "warning"));
  }
  strip.replaceChildren(...badges);
}

function showToast(message) {
  const toast = $("toast");
  toast.textContent = message;
  toast.classList.add("visible");
  window.setTimeout(() => toast.classList.remove("visible"), 4000);
}

function showError(error) {
  if (error.status === 401) {
    showLogin();
    return;
  }
  showToast(error.message || "Unexpected error");
}

function selectControl(values, value, field, id) {
  const select = document.createElement("select");
  select.dataset.field = field;
  select.dataset.id = id;
  const row = state.files.find((item) => item.id === id);
  select.setAttribute("aria-label", `${field.replace("_", " ")} for ${row ? row.filename : "file"}`);
  for (const [optionValue, label] of values) {
    const option = document.createElement("option");
    option.value = optionValue;
    option.textContent = label;
    option.selected = String(value ?? "") === optionValue;
    select.append(option);
  }
  select.addEventListener("change", () => updateStagingItem(id, field, select.value));
  return select;
}

async function addFiles(fileList) {
  const rows = Array.from(fileList).map((file) => ({
    id: crypto.randomUUID(),
    file,
    filename: file.name,
    relative_path: file.webkitRelativePath || file.name,
    kind: "general",
    subject: "",
    grade: "",
    lang: "",
    lab_number: "",
    ocr: state.ocrDefault,
    ocrOverridden: false,
    previewErrors: [],
  }));
  state.files.push(...rows);
  try {
    await previewPaths(rows);
  } catch (error) {
    rows.forEach((row) => { row.previewErrors = [error.message]; });
  }
  renderStaging();
}

async function previewPaths(rows) {
  const payload = await request("/api/admin/ingestion/preview", {
    method: "POST",
    json: { paths: rows.map((item) => item.relative_path) },
  });
  payload.items.forEach((preview, index) => {
    const row = rows[index];
    row.previewErrors = (preview.errors || []).filter(
      (error) => !error.startsWith("Duplicate document identity:"),
    );
    if (!preview.metadata) return;
    row.kind = preview.metadata.doc_type;
    row.subject = preview.metadata.subject || "";
    row.grade = preview.metadata.grade ? String(preview.metadata.grade) : "";
    row.lang = preview.metadata.lang || "";
    row.lab_number = preview.metadata.lab_number ? String(preview.metadata.lab_number) : "";
  });
}

function validateRow(row) {
  const errors = [...row.previewErrors];
  if (![".pdf", ".docx", ".epub", ".txt", ".md"].some((suffix) => row.filename.toLowerCase().endsWith(suffix))) {
    errors.push("Unsupported file type");
  }
  if (row.kind !== "general") {
    if (!row.subject) errors.push("Subject is required");
    if (!row.grade) errors.push("Grade is required");
    if (!row.lang) errors.push("Language is required");
  }
  if (row.kind === "lab_instruction" && !row.lab_number) errors.push("Lab number is required");
  if (row.kind === "textbook" && row.lab_number) errors.push("Textbooks cannot have a lab number");
  const identity = stagingIdentity(row);
  // ponytail: O(n^2) over at most 100 staged files; index only if that limit grows.
  if (identity && state.files.filter((item) => stagingIdentity(item) === identity).length > 1) {
    errors.push("Duplicate document identity");
  }
  return errors;
}

function stagingIdentity(row) {
  if (row.kind === "general") return `general:${row.relative_path}`;
  if (!row.subject || !row.grade || !row.lang) return null;
  if (row.kind === "textbook") {
    return `textbook:${row.subject}:${row.grade}:${row.lang}:${row.filename}`;
  }
  if (!row.lab_number) return null;
  return `lab_instruction:${row.subject}:${row.grade}:${row.lang}:${Number(row.lab_number)}`;
}

function cell(content) {
  const td = document.createElement("td");
  if (content instanceof Node) td.append(content);
  else td.textContent = String(content ?? "");
  return td;
}

function renderStaging() {
  const body = $("stagingTable").tBodies[0];
  body.replaceChildren();
  let valid = state.files.length > 0;
  for (const row of state.files) {
    const tr = document.createElement("tr");
    const errors = validateRow(row);
    valid = valid && errors.length === 0;
    tr.append(
      cell(row.relative_path),
      cell(selectControl([["general", "General"], ["textbook", "Textbook"], ["lab_instruction", "Lab instruction"]], row.kind, "kind", row.id)),
      cell(selectControl([["", "None"], ["physics", "Physics"], ["chemistry", "Chemistry"], ["biology", "Biology"]], row.subject, "subject", row.id)),
      cell(selectControl([["", "None"], ...[7, 8, 9, 10, 11].map((grade) => [String(grade), String(grade)])], row.grade, "grade", row.id)),
      cell(selectControl([["", "None"], ["ru", "ru"], ["kk", "kk"], ["en", "en"]], row.lang, "lang", row.id)),
      cell(selectControl([["", "None"], ...Array.from({ length: 99 }, (_, index) => [String(index + 1), String(index + 1)])], row.lab_number, "lab_number", row.id)),
    );
    const ocr = document.createElement("input");
    ocr.type = "checkbox";
    ocr.checked = row.ocr;
    ocr.setAttribute("aria-label", `OCR ${row.filename}`);
    ocr.addEventListener("change", () => updateStagingItem(row.id, "ocr", ocr.checked));
    tr.append(cell(ocr), cell(errors.length ? errors.join("; ") : "Ready"));
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "Remove";
    remove.setAttribute("aria-label", `Remove ${row.filename}`);
    remove.addEventListener("click", () => {
      state.files = state.files.filter((item) => item.id !== row.id);
      renderStaging();
    });
    tr.append(cell(remove));
    body.append(tr);
  }
  $("queueUpload").disabled = !valid;
}

function updateStagingItem(id, field, value) {
  const row = state.files.find((item) => item.id === id);
  if (!row) return;
  row[field] = value;
  if (field === "ocr") row.ocrOverridden = true;
  row.previewErrors = [];
  if (field === "kind" && value !== "lab_instruction") row.lab_number = "";
  if (field === "kind" && value === "general") {
    row.subject = "";
    row.grade = "";
    row.lang = "";
  }
  renderStaging();
}

function toManifestItem(row) {
  const structured = row.kind !== "general";
  return {
    filename: row.filename,
    relative_path: row.relative_path,
    doc_type: structured ? row.kind : null,
    subject: structured ? row.subject : null,
    grade: structured ? Number(row.grade) : null,
    lang: structured ? row.lang : null,
    lab_number: row.kind === "lab_instruction" ? Number(row.lab_number) : null,
    ocr: row.ocr,
  };
}

async function queueUpload() {
  if (!state.files.length || state.files.some((row) => validateRow(row).length)) return;
  const form = new FormData();
  for (const item of state.files) form.append("files", item.file, item.file.name);
  form.append("manifest", JSON.stringify(state.files.map(toManifestItem)));
  const job = await request("/api/admin/ingestion/jobs/upload", { method: "POST", body: form });
  state.files = [];
  renderStaging();
  selectView("jobs");
  await loadJob(job.id);
  await loadJobs();
}

async function previewCorpus() {
  state.corpusPreview = await request("/api/admin/ingestion/corpus/preview", {
    method: "POST",
    json: corpusOptions(),
  });
  renderCorpusPreview();
}

async function queueCorpus() {
  if (!state.corpusPreview) return;
  const options = corpusOptions();
  if (options.prune && !await confirmChange("Queue a full-root ingest that may prune missing corpus documents?")) return;
  const job = await request("/api/admin/ingestion/jobs/corpus", {
    method: "POST",
    json: options,
  });
  selectView("jobs");
  await loadJob(job.id);
  await loadJobs();
}

async function loadJobs() {
  const payload = await request("/api/admin/ingestion/jobs?limit=100");
  state.jobs = payload.jobs;
  renderJobs();
}

async function loadDocuments(existingStatus = null) {
  const [status, payload] = await Promise.all([
    existingStatus || request("/api/admin/corpus_status"),
    request("/api/admin/documents"),
  ]);
  state.documents = payload.documents;
  renderCorpusSummary(status);
  renderDocuments();
}

function corpusOptions() {
  return {
    subtree: $("corpusSubtree").value.trim(),
    ocr: $("corpusOcr").checked,
    prune: $("corpusPrune").checked,
  };
}

function renderCorpusPreview() {
  const container = $("corpusPreview");
  container.replaceChildren();
  if (!state.corpusPreview) {
    $("queueCorpus").disabled = true;
    return;
  }
  const summary = document.createElement("p");
  const typeCounts = Object.entries(state.corpusPreview.counts_by_type || {})
    .map(([name, count]) => `${name} ${count}`)
    .join(", ");
  const languageCounts = Object.entries(state.corpusPreview.counts_by_language || {})
    .map(([name, count]) => `${name} ${count}`)
    .join(", ");
  summary.textContent = `${state.corpusPreview.recognized} recognized of ${state.corpusPreview.total}; ${state.corpusPreview.skipped.length} skipped; ${state.corpusPreview.prunable} prunable; types ${typeCounts || "none"}; languages ${languageCounts || "none"}`;
  container.append(summary);
  if ((state.corpusPreview.duplicate_lab_ids || []).length) {
    const duplicates = document.createElement("p");
    duplicates.className = "error";
    duplicates.textContent = `Duplicate labs: ${state.corpusPreview.duplicate_lab_ids.join(", ")}`;
    container.append(duplicates);
  }
  if (state.corpusPreview.skipped.length) {
    const list = document.createElement("ul");
    for (const item of state.corpusPreview.skipped) {
      const entry = document.createElement("li");
      entry.textContent = `${item.source}: ${item.error}`;
      list.append(entry);
    }
    container.append(list);
  }
  $("queueCorpus").disabled = state.corpusPreview.recognized === 0;
}

function clearJobDetails() {
  state.selectedJobId = null;
  const details = $("jobDetails");
  details.replaceChildren();
  details.hidden = true;
}

function commandButton(label, action, danger = false) {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = label;
  if (danger) button.classList.add("danger");
  button.addEventListener("click", action);
  return button;
}

async function confirmChange(message) {
  const dialog = $("confirmDialog");
  $("confirmMessage").textContent = message;
  dialog.showModal();
  return new Promise((resolve) => {
    dialog.addEventListener("close", () => resolve(dialog.returnValue === "confirm"), { once: true });
  });
}

async function loadJob(id, options = {}) {
  let job;
  try {
    job = await request(`/api/admin/ingestion/jobs/${id}`);
  } catch (error) {
    if (options.quietMissing && error.status === 404) {
      clearJobDetails();
      return null;
    }
    throw error;
  }
  state.selectedJobId = job.id;
  const details = $("jobDetails");
  details.replaceChildren();
  details.hidden = false;
  const heading = document.createElement("h2");
  heading.textContent = `Job ${job.id}`;
  const summary = document.createElement("p");
  summary.textContent = `${job.status}; ${job.completed_items}/${job.total_items} completed; ${job.failed_items} failed; ${job.skipped_items} skipped`;
  const timestamps = document.createElement("p");
  timestamps.textContent = `Created ${job.created_at}; started ${job.started_at || "not started"}; finished ${job.finished_at || "not finished"}`;
  const table = document.createElement("table");
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  for (const label of ["File", "Status", "Stage", "Chunks", "Error"]) headRow.append(cell(label));
  head.append(headRow);
  const body = document.createElement("tbody");
  for (const item of job.items) {
    const row = document.createElement("tr");
    row.append(cell(item.relative_path || item.filename), cell(item.status), cell(item.stage), cell(item.chunks ?? ""), cell(item.error || ""));
    body.append(row);
  }
  table.append(head, body);
  const tableWrap = document.createElement("div");
  tableWrap.className = "table-wrap";
  tableWrap.append(table);
  details.append(heading, summary, timestamps);
  for (const [message, className] of [[job.error, "error"], [job.warning, ""]]) {
    if (!message) continue;
    const notice = document.createElement("p");
    notice.className = className;
    notice.textContent = message;
    details.append(notice);
  }
  details.append(tableWrap);
  return job;
}

async function refreshSelectedJob() {
  if (state.selectedJobId) await loadJob(state.selectedJobId, { quietMissing: true });
}

async function cancelJob(id) {
  if (!await confirmChange("Cancel this ingestion job?")) return;
  await request(`/api/admin/ingestion/jobs/${id}/cancel`, { method: "POST" });
  await loadJobs();
  await loadJob(id, { quietMissing: true });
}

async function retryJob(id) {
  const job = await request(`/api/admin/ingestion/jobs/${id}/retry`, { method: "POST" });
  await loadJobs();
  await loadJob(job.id);
}

async function deleteJob(id) {
  if (!await confirmChange("Delete this job history and its retained upload files?")) return;
  await request(`/api/admin/ingestion/jobs/${id}`, { method: "DELETE" });
  clearJobDetails();
  await loadJobs();
}

function renderJobs() {
  const body = $("jobsTable").tBodies[0];
  body.replaceChildren();
  for (const job of state.jobs) {
    const row = document.createElement("tr");
    const actions = document.createElement("div");
    actions.className = "command-row";
    actions.append(commandButton("Details", () => loadJob(job.id).catch(showError)));
    if (["queued", "running"].includes(job.status)) {
      actions.append(commandButton("Cancel", () => cancelJob(job.id).catch(showError), true));
    }
    if (["failed", "partial", "cancelled"].includes(job.status)) {
      actions.append(commandButton("Retry", () => retryJob(job.id).catch(showError)));
    }
    if (["completed", "partial", "failed", "cancelled"].includes(job.status)) {
      actions.append(commandButton("Delete", () => deleteJob(job.id).catch(showError), true));
    }
    row.append(
      cell(job.status),
      cell(job.kind),
      cell(`${job.completed_items}/${job.total_items}`),
      cell([job.current_item, job.current_stage].filter(Boolean).join("; ")),
      cell(job.created_at),
      cell(actions),
    );
    body.append(row);
  }
}

function renderCorpusSummary(status) {
  $("corpusSummary").textContent = `${status.documents || 0} documents; ${status.points || 0} chunks; ${status.status}`;
}

async function deleteDocument(fileId, filename) {
  if (!await confirmChange(`Delete ${filename} from the knowledge base?`)) return;
  await request(`/api/admin/documents/${encodeURIComponent(fileId)}`, { method: "DELETE" });
  await loadDocuments();
}

function renderDocuments() {
  const search = $("documentSearch").value.trim().toLowerCase();
  const type = $("documentType").value;
  const subject = $("documentSubject").value;
  const grade = $("documentGrade").value;
  const lang = $("documentLanguage").value;
  const filtered = state.documents.filter((document) =>
    (!search || (document.filename || "").toLowerCase().includes(search))
    && (!type || document.doc_type === type)
    && (!subject || document.subject === subject)
    && (!grade || String(document.grade || "") === grade)
    && (!lang || document.lang === lang)
  );
  const body = $("documentsTable").tBodies[0];
  body.replaceChildren();
  for (const item of filtered) {
    const remove = commandButton(
      "Delete",
      () => deleteDocument(item.file_id, item.filename || item.file_id).catch(showError),
      true,
    );
    remove.setAttribute("aria-label", `Delete ${item.filename || item.file_id}`);
    const row = document.createElement("tr");
    row.append(
      cell(item.filename || ""),
      cell(item.doc_type || item.source_type || "general"),
      cell(item.subject || ""),
      cell(item.grade || ""),
      cell(item.lang || ""),
      cell(item.chunks || 0),
      cell(remove),
    );
    body.append(row);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  $("loginForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    $("loginError").textContent = "";
    try {
      const result = await request("/auth/login", {
        method: "POST",
        json: { username: $("username").value, password: $("password").value },
      });
      state.csrf = result.csrf_token;
      $("password").value = "";
      showApp();
      await refreshAll();
    } catch (error) {
      $("loginError").textContent = error.message;
    }
  });
  $("logoutButton").addEventListener("click", async () => {
    try {
      await request("/auth/logout", { method: "POST" });
    } catch {
      // Session may already be expired server-side; local reset still matters.
    } finally {
      showLogin();
    }
  });
  document.querySelectorAll("[data-view]").forEach((button) => button.addEventListener("click", () => selectView(button.dataset.view)));
  document.querySelectorAll("[data-source]").forEach((button) => button.addEventListener("click", () => selectSource(button.dataset.source)));
  $("fileInput").addEventListener("change", (event) => addFiles(event.target.files).catch(showError));
  $("folderInput").addEventListener("change", (event) => addFiles(event.target.files).catch(showError));
  $("clearStaging").addEventListener("click", () => { state.files = []; renderStaging(); });
  $("queueUpload").addEventListener("click", () => queueUpload().catch(showError));
  $("previewCorpus").addEventListener("click", () => previewCorpus().catch(showError));
  $("queueCorpus").addEventListener("click", () => queueCorpus().catch(showError));
  $("corpusOcr").addEventListener("change", () => {
    state.corpusOcrOverridden = true;
  });
  $("corpusSubtree").addEventListener("input", () => {
    const hasSubtree = Boolean($("corpusSubtree").value.trim());
    $("corpusPrune").checked = false;
    $("corpusPrune").disabled = hasSubtree;
    state.corpusPreview = null;
    renderCorpusPreview();
    $("queueCorpus").disabled = true;
  });
  for (const id of ["documentSearch", "documentType", "documentSubject", "documentGrade", "documentLanguage"]) {
    $(id).addEventListener(id === "documentSearch" ? "input" : "change", renderDocuments);
  }
  bootstrap();
});
