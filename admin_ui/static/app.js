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

const labels = {
  status: {
    queued: "В очереди",
    running: "Выполняется",
    completed: "Завершено",
    partial: "Частично",
    failed: "Ошибка",
    cancelled: "Отменено",
    pending: "Ожидает",
    skipped: "Пропущено",
    ready: "Готово",
    empty: "Пусто",
  },
  kind: {
    upload: "Загрузка файлов",
    corpus: "Корпус сервера",
    general: "Обычный документ",
    textbook: "Учебник",
    lab_instruction: "Лабораторная работа",
  },
  stage: {
    extracting: "Извлечение текста",
    parsing: "Разбор",
    chunking: "Разбиение",
    embedding: "Создание векторов",
    indexing: "Индексация",
    uploading: "Загрузка",
    finalizing: "Завершение",
    done: "Готово",
  },
  subject: {
    physics: "Физика",
    chemistry: "Химия",
    biology: "Биология",
  },
  language: {
    ru: "Русский",
    kk: "Казахский",
    en: "Английский",
  },
  field: {
    kind: "Тип",
    subject: "Предмет",
    grade: "Класс",
    lang: "Язык",
    lab_number: "Номер лабораторной работы",
  },
};

const messageLabels = {
  "Invalid credentials": "Неверный логин или пароль",
  "Too many login attempts": "Слишком много попыток входа. Повторите позже.",
  "Authentication required": "Требуется вход",
  "Invalid CSRF token": "Сессия устарела. Войдите снова.",
  "Backend admin API unavailable": "Сервер администрирования недоступен",
  "Cancelled by administrator": "Отменено администратором",
  "No usable text extracted": "Не удалось извлечь текст",
  "Interrupted while processing": "Обработка была прервана",
  "Unrecognised corpus path": "Путь в корпусе не распознан",
  "Unrecognized corpus path": "Путь в корпусе не распознан",
  "Job not found": "Задание не найдено",
  "Document not found": "Документ не найден",
};

function label(group, value) {
  return labels[group]?.[value] || value || "";
}

function translateMessage(message) {
  return messageLabels[message] || message || "";
}

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
    const error = new Error(detail || `Запрос завершился с ошибкой ${response.status}`);
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
    statusBadge("Сервер доступен", "success"),
    statusBadge(`Qdrant: ${label("status", corpusStatus.status)}`, qdrantReady ? "success" : "warning"),
    statusBadge(status.worker.online ? "Обработчик доступен" : "Обработчик недоступен", status.worker.online ? "success" : "danger"),
    statusBadge(`В очереди: ${status.queue.queued || 0}`, status.queue.queued ? "warning" : ""),
    statusBadge(`Выполняется: ${status.queue.running || 0}`, status.queue.running ? "warning" : ""),
  ];
  if (active) {
    const activity = [active.current_item, label("stage", active.current_stage)].filter(Boolean).join(": ");
    badges.push(statusBadge(`Активно: ${activity || active.id.slice(0, 8)}`, "warning"));
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
  showToast(translateMessage(error.message) || "Непредвиденная ошибка");
}

function selectControl(values, value, field, id) {
  const select = document.createElement("select");
  select.dataset.field = field;
  select.dataset.id = id;
  const row = state.files.find((item) => item.id === id);
  select.setAttribute("aria-label", `${label("field", field)}: ${row ? row.filename : "файл"}`);
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
  const errors = row.previewErrors.map(translateMessage);
  if (![".pdf", ".docx", ".epub", ".txt", ".md"].some((suffix) => row.filename.toLowerCase().endsWith(suffix))) {
    errors.push("Неподдерживаемый тип файла");
  }
  if (row.kind !== "general") {
    if (!row.subject) errors.push("Укажите предмет");
    if (!row.grade) errors.push("Укажите класс");
    if (!row.lang) errors.push("Укажите язык");
  }
  if (row.kind === "lab_instruction" && !row.lab_number) errors.push("Укажите номер лабораторной работы");
  if (row.kind === "textbook" && row.lab_number) errors.push("Для учебника нельзя указывать номер лабораторной работы");
  const identity = stagingIdentity(row);
  // ponytail: O(n^2) over at most 100 staged files; index only if that limit grows.
  if (identity && state.files.filter((item) => stagingIdentity(item) === identity).length > 1) {
    errors.push("Документ с такими реквизитами уже добавлен");
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
      cell(selectControl([["general", "Обычный документ"], ["textbook", "Учебник"], ["lab_instruction", "Лабораторная работа"]], row.kind, "kind", row.id)),
      cell(selectControl([["", "Не указано"], ["physics", "Физика"], ["chemistry", "Химия"], ["biology", "Биология"]], row.subject, "subject", row.id)),
      cell(selectControl([["", "Не указано"], ...[7, 8, 9, 10, 11].map((grade) => [String(grade), String(grade)])], row.grade, "grade", row.id)),
      cell(selectControl([["", "Не указано"], ["ru", "Русский"], ["kk", "Казахский"], ["en", "Английский"]], row.lang, "lang", row.id)),
      cell(selectControl([["", "Не указано"], ...Array.from({ length: 99 }, (_, index) => [String(index + 1), String(index + 1)])], row.lab_number, "lab_number", row.id)),
    );
    const ocr = document.createElement("input");
    ocr.type = "checkbox";
    ocr.checked = row.ocr;
    ocr.setAttribute("aria-label", `OCR: ${row.filename}`);
    ocr.addEventListener("change", () => updateStagingItem(row.id, "ocr", ocr.checked));
    tr.append(cell(ocr), cell(errors.length ? errors.join("; ") : "Готово"));
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "Удалить";
    remove.setAttribute("aria-label", `Удалить ${row.filename}`);
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
  if (options.prune && !await confirmChange("Поставить в очередь полное сканирование с удалением отсутствующих документов?")) return;
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
    .map(([name, count]) => `${label("kind", name)}: ${count}`)
    .join(", ");
  const languageCounts = Object.entries(state.corpusPreview.counts_by_language || {})
    .map(([name, count]) => `${label("language", name)}: ${count}`)
    .join(", ");
  summary.textContent = `Распознано ${state.corpusPreview.recognized} из ${state.corpusPreview.total}; пропущено: ${state.corpusPreview.skipped.length}; к удалению: ${state.corpusPreview.prunable}; типы: ${typeCounts || "нет"}; языки: ${languageCounts || "нет"}`;
  container.append(summary);
  if ((state.corpusPreview.duplicate_lab_ids || []).length) {
    const duplicates = document.createElement("p");
    duplicates.className = "error";
    duplicates.textContent = `Повторяющиеся лабораторные работы: ${state.corpusPreview.duplicate_lab_ids.join(", ")}`;
    container.append(duplicates);
  }
  if (state.corpusPreview.skipped.length) {
    const list = document.createElement("ul");
    for (const item of state.corpusPreview.skipped) {
      const entry = document.createElement("li");
      entry.textContent = `${item.source}: ${translateMessage(item.error)}`;
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
  heading.textContent = `Задание ${job.id}`;
  const summary = document.createElement("p");
  summary.textContent = `${label("status", job.status)}; выполнено ${job.completed_items}/${job.total_items}; с ошибкой: ${job.failed_items}; пропущено: ${job.skipped_items}`;
  const timestamps = document.createElement("p");
  timestamps.textContent = `Создано: ${job.created_at}; запущено: ${job.started_at || "не запущено"}; завершено: ${job.finished_at || "не завершено"}`;
  const table = document.createElement("table");
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  for (const headingLabel of ["Файл", "Статус", "Этап", "Фрагменты", "Ошибка"]) headRow.append(cell(headingLabel));
  head.append(headRow);
  const body = document.createElement("tbody");
  for (const item of job.items) {
    const row = document.createElement("tr");
    row.append(cell(item.relative_path || item.filename), cell(label("status", item.status)), cell(label("stage", item.stage)), cell(item.chunks ?? ""), cell(translateMessage(item.error)));
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
    notice.textContent = translateMessage(message);
    details.append(notice);
  }
  details.append(tableWrap);
  return job;
}

async function refreshSelectedJob() {
  if (state.selectedJobId) await loadJob(state.selectedJobId, { quietMissing: true });
}

async function cancelJob(id) {
  if (!await confirmChange("Отменить это задание на загрузку?")) return;
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
  if (!await confirmChange("Удалить историю задания и сохранённые загруженные файлы?")) return;
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
    actions.append(commandButton("Детали", () => loadJob(job.id).catch(showError)));
    if (["queued", "running"].includes(job.status)) {
      actions.append(commandButton("Отменить", () => cancelJob(job.id).catch(showError), true));
    }
    if (["failed", "partial", "cancelled"].includes(job.status)) {
      actions.append(commandButton("Повторить", () => retryJob(job.id).catch(showError)));
    }
    if (["completed", "partial", "failed", "cancelled"].includes(job.status)) {
      actions.append(commandButton("Удалить", () => deleteJob(job.id).catch(showError), true));
    }
    row.append(
      cell(label("status", job.status)),
      cell(label("kind", job.kind)),
      cell(`${job.completed_items}/${job.total_items}`),
      cell([job.current_item, label("stage", job.current_stage)].filter(Boolean).join("; ")),
      cell(job.created_at),
      cell(actions),
    );
    body.append(row);
  }
}

function renderCorpusSummary(status) {
  $("corpusSummary").textContent = `${status.documents || 0} документов; ${status.points || 0} фрагментов; ${label("status", status.status)}`;
}

async function deleteDocument(fileId, filename) {
  if (!await confirmChange(`Удалить ${filename} из базы знаний?`)) return;
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
      "Удалить",
      () => deleteDocument(item.file_id, item.filename || item.file_id).catch(showError),
      true,
    );
    remove.setAttribute("aria-label", `Удалить ${item.filename || item.file_id}`);
    const row = document.createElement("tr");
    row.append(
      cell(item.filename || ""),
      cell(label("kind", item.doc_type || item.source_type || "general")),
      cell(label("subject", item.subject)),
      cell(item.grade || ""),
      cell(label("language", item.lang)),
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
      $("loginError").textContent = translateMessage(error.message);
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
