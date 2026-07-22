const testState = {
  scenariosLoaded: false,
  sttBlob: null,
  voiceBlob: null,
  sttPreviewUrl: null,
  voicePreviewUrl: null,
  ttsUrl: null,
  recorderResets: [],
};

function escapeTest(value) {
  return String(value ?? "").replace(/[&<>]/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
  })[character]);
}

function testMeta(status, milliseconds, extra = "") {
  const className = status >= 200 && status < 300 ? "success" : "danger";
  return `<div class="test-meta"><span class="badge ${className}">HTTP ${status}</span><span class="badge">${milliseconds} мс</span>${extra}</div>`;
}

function testRaw(payload, label = "Ответ JSON") {
  return `<details><summary>${label}</summary><pre>${escapeTest(JSON.stringify(payload, null, 2))}</pre></details>`;
}

function testCitations(citations, primarySource) {
  if (!citations?.length) return "";
  const primary = primarySource ? `; основной: ${escapeTest(JSON.stringify(primarySource))}` : "";
  return `<details><summary>Источники: ${citations.length}${primary}</summary>${citations.map((citation) => `<div class="test-citation">${escapeTest(JSON.stringify(citation))}</div>`).join("")}</details>`;
}

function renderTestError(output, error) {
  const detail = error.payload ?? error.message ?? error;
  const text = typeof detail === "string" ? detail : JSON.stringify(detail, null, 2);
  output.innerHTML = `${error.status ? testMeta(error.status, 0) : ""}<pre class="error">${escapeTest(text)}</pre>`;
}

function renderTestPending(output, message) {
  output.innerHTML = `<p class="muted">${escapeTest(message)}</p>`;
}

async function testFetch(path, options = {}) {
  const method = (options.method || "GET").toUpperCase();
  const headers = new Headers(options.headers || {});
  const fetchOptions = { ...options };
  delete fetchOptions.json;
  if (method === "POST") headers.set("X-CSRF-Token", state.csrf);
  if (options.json !== undefined) {
    headers.set("Content-Type", "application/json");
    fetchOptions.body = JSON.stringify(options.json);
  }
  const started = performance.now();
  const response = await fetch(`/api/test/${path}`, {
    ...fetchOptions,
    method,
    headers,
    credentials: "same-origin",
  });
  return { response, milliseconds: Math.round(performance.now() - started) };
}

async function testPayload(response) {
  const contentType = response.headers.get("content-type") || "";
  return contentType.includes("application/json") ? response.json() : response.text();
}

function testFailure(status, payload) {
  const detail = typeof payload === "object" ? payload.detail || payload : payload;
  const error = new Error(typeof detail === "string" ? detail : `HTTP ${status}`);
  error.status = status;
  error.payload = detail;
  return error;
}

function selectTestView(name) {
  document.querySelectorAll("[data-test-view]").forEach((button) => {
    button.setAttribute("aria-selected", String(button.dataset.testView === name));
  });
  document.querySelectorAll("[data-test-panel]").forEach((panel) => {
    panel.hidden = panel.dataset.testPanel !== name;
  });
}

function resetTestConsole() {
  for (const reset of testState.recorderResets) reset();
  for (const field of ["sttPreviewUrl", "voicePreviewUrl", "ttsUrl"]) {
    if (testState[field]) URL.revokeObjectURL(testState[field]);
    testState[field] = null;
  }
  testState.scenariosLoaded = false;
  testState.sttBlob = null;
  testState.voiceBlob = null;
  for (const control of document.querySelectorAll("#testingView input, #testingView textarea, #testingView select")) {
    if (control instanceof HTMLInputElement && control.type === "file") {
      control.value = "";
    } else if (control instanceof HTMLInputElement && ["checkbox", "radio"].includes(control.type)) {
      control.checked = control.defaultChecked;
    } else if (control instanceof HTMLSelectElement) {
      const defaultIndex = Array.from(control.options).findIndex((option) => option.defaultSelected);
      control.selectedIndex = defaultIndex >= 0 ? defaultIndex : 0;
    } else {
      control.value = control.defaultValue;
    }
  }
  document.querySelectorAll("#testingView .test-output, #testSttPreview, #testVoicePreview").forEach((element) => element.replaceChildren());
  $("scenarioList").replaceChildren();
  $("testSttRecordStatus").textContent = "";
  $("testVoiceRecordStatus").textContent = "";
  $("testSttRecordStatus").className = "muted";
  $("testVoiceRecordStatus").className = "muted";
  $("testSttFileName").textContent = "Файл не выбран";
  $("testVoiceFileName").textContent = "Файл не выбран";
  selectTestView("health");
}

function testLab(prefix) {
  const subject = $(`${prefix}Subject`).value;
  const grade = $(`${prefix}Grade`).value;
  if (!subject || !grade) return null;
  const lab = { subject, grade: Number(grade) };
  const lang = $(`${prefix}LabLang`).value;
  const labNumber = $(`${prefix}LabNumber`).value;
  if (lang) lab.lang = lang;
  if (labNumber) lab.lab_number = Number(labNumber);
  return lab;
}

async function runTestHealth() {
  const output = $("testHealthOutput");
  renderTestPending(output, "Проверка API...");
  const { response, milliseconds } = await testFetch("health");
  const payload = await testPayload(response);
  if (!response.ok) throw testFailure(response.status, payload);
  output.innerHTML = testMeta(response.status, milliseconds) + testRaw(payload);
}

async function runTestReady() {
  const output = $("testHealthOutput");
  renderTestPending(output, "Проверка готовности...");
  const { response, milliseconds } = await testFetch("ready");
  const payload = await testPayload(response);
  if (!response.ok) throw testFailure(response.status, payload);
  output.innerHTML = testMeta(response.status, milliseconds) + testRaw(payload);
}

async function loadTestScenarios(showResult = false) {
  const started = performance.now();
  const payload = await request("/api/admin/scenarios");
  const list = $("scenarioList");
  list.replaceChildren();
  for (const scenario of payload.scenarios || []) {
    const option = document.createElement("option");
    option.value = typeof scenario === "string" ? scenario : scenario.scenario_id || scenario.id || "";
    if (option.value) list.append(option);
  }
  testState.scenariosLoaded = true;
  if (showResult) {
    $("testHealthOutput").innerHTML = testMeta(200, Math.round(performance.now() - started)) + testRaw(payload);
  }
}

async function runTestAsk() {
  const output = $("testAskOutput");
  renderTestPending(output, "Формируется ответ...");
  const body = { query: $("testAskQuery").value };
  if ($("testAskResponseLang").value) body.language = $("testAskResponseLang").value;
  if ($("testAskScenario").value) body.scenario_id = $("testAskScenario").value;
  if ($("testAskMaxTokens").value) body.max_tokens = Number($("testAskMaxTokens").value);
  const lab = testLab("testAsk");
  if (lab) body.lab = lab;
  const { response, milliseconds } = await testFetch("ask", { method: "POST", json: body });
  const payload = await testPayload(response);
  if (!response.ok) throw testFailure(response.status, payload);
  const usage = payload.usage ? `<span class="badge">${escapeTest(JSON.stringify(payload.usage))}</span>` : "";
  output.innerHTML = testMeta(response.status, milliseconds, usage)
    + `<div class="test-answer">${escapeTest(payload.answer)}</div>`
    + testCitations(payload.citations, payload.primary_source)
    + testRaw(payload);
}

async function runTestChat() {
  const output = $("testChatOutput");
  let messages;
  try {
    messages = JSON.parse($("testChatMessages").value);
  } catch (error) {
    throw new Error(`Некорректный JSON: ${error.message}`);
  }
  const stream = $("testChatStream").checked;
  const body = { messages, stream };
  if ($("testChatLanguage").value) body.language = $("testChatLanguage").value;
  if ($("testChatScenario").value) body.scenario_id = $("testChatScenario").value;
  if ($("testChatMaxTokens").value) body.max_tokens = Number($("testChatMaxTokens").value);
  renderTestPending(output, stream ? "Получение потока..." : "Формируется ответ...");
  const started = performance.now();
  const { response, milliseconds } = await testFetch("v1/chat/completions", { method: "POST", json: body });

  if (!stream) {
    const payload = await testPayload(response);
    if (!response.ok) throw testFailure(response.status, payload);
    const answer = payload.choices?.[0]?.message?.content || "";
    const metadata = payload.metadata || {};
    output.innerHTML = testMeta(response.status, milliseconds)
      + `<div class="test-answer">${escapeTest(answer)}</div>`
      + testCitations(metadata.citations, metadata.primary_source)
      + testRaw(payload);
    return;
  }

  if (!response.ok) throw testFailure(response.status, await testPayload(response));
  output.innerHTML = testMeta(response.status, milliseconds, '<span class="badge warning">Поток</span>')
    + '<div id="testChatStreamAnswer" class="test-answer"></div><div id="testChatStreamMeta"></div>';
  const answer = $("testChatStreamAnswer");
  const metadata = $("testChatStreamMeta");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let text = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split(/\r?\n\r?\n/);
    buffer = frames.pop();
    for (const frame of frames) {
      const line = frame.split(/\r?\n/).find((item) => item.startsWith("data:"));
      if (!line) continue;
      const data = line.slice(5).trim();
      if (!data || data === "[DONE]") continue;
      try {
        const event = JSON.parse(data);
        const delta = event.choices?.[0]?.delta?.content;
        if (delta) {
          text += delta;
          answer.textContent = text;
        } else if (event.metadata) {
          metadata.innerHTML = testCitations(event.metadata.citations, event.metadata.primary_source);
        } else if (event.error) {
          metadata.innerHTML += `<pre class="error">${escapeTest(JSON.stringify(event.error, null, 2))}</pre>`;
        }
      } catch {
        // Ignore non-JSON SSE frames.
      }
    }
  }
  output.querySelector(".test-meta").innerHTML = `<span class="badge success">HTTP ${response.status}</span><span class="badge">${Math.round(performance.now() - started)} мс</span><span class="badge success">Готово</span>`;
}

async function runTestHint() {
  const output = $("testHintOutput");
  renderTestPending(output, "Формируется подсказка...");
  const body = {
    hint_text: $("testHintText").value,
    hint_level: Number($("testHintLevel").value),
  };
  if ($("testHintLanguage").value) body.language = $("testHintLanguage").value;
  if ($("testHintScenario").value) body.scenario_id = $("testHintScenario").value;
  const { response, milliseconds } = await testFetch("hint", { method: "POST", json: body });
  const payload = await testPayload(response);
  if (!response.ok) throw testFailure(response.status, payload);
  output.innerHTML = testMeta(response.status, milliseconds)
    + `<div class="test-answer">${escapeTest(payload.hint)}</div>`
    + testRaw(payload);
}

function renderRecordedAudio(previewId, blob, urlField) {
  if (testState[urlField]) URL.revokeObjectURL(testState[urlField]);
  testState[urlField] = URL.createObjectURL(blob);
  const audio = document.createElement("audio");
  audio.controls = true;
  audio.src = testState[urlField];
  $(previewId).replaceChildren(audio);
}

function makeTestRecorder(buttonId, statusId, previewId, blobField, urlField) {
  let recorder = null;
  let stream = null;
  let chunks = [];
  let discard = false;
  const reset = () => {
    discard = recorder !== null;
    if (recorder && recorder.state !== "inactive") {
      try {
        recorder.stop();
      } catch {
        // The stream may already have stopped.
      }
    }
    stream?.getTracks().forEach((track) => track.stop());
    if (!recorder) chunks = [];
    $(buttonId).textContent = "Начать запись";
    $(buttonId).classList.remove("recording");
  };
  const toggle = async () => {
    const button = $(buttonId);
    const status = $(statusId);
    if (recorder?.state === "recording") {
      recorder.stop();
      return;
    }
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recorder = new MediaRecorder(stream);
    } catch (error) {
      stream?.getTracks().forEach((track) => track.stop());
      stream = null;
      recorder = null;
      status.textContent = `Ошибка микрофона: ${error.message}`;
      status.className = "error";
      return;
    }
    chunks = [];
    recorder.addEventListener("dataavailable", (event) => {
      if (event.data.size) chunks.push(event.data);
    });
    recorder.addEventListener("stop", () => {
      const mimeType = recorder?.mimeType || "audio/webm";
      stream?.getTracks().forEach((track) => track.stop());
      stream = null;
      recorder = null;
      if (discard) {
        discard = false;
        chunks = [];
        return;
      }
      const blob = new Blob(chunks, { type: mimeType });
      testState[blobField] = blob;
      renderRecordedAudio(previewId, blob, urlField);
      status.textContent = `Записано ${(blob.size / 1024).toFixed(1)} КБ`;
      status.className = "muted";
      button.textContent = "Начать запись";
      button.classList.remove("recording");
    });
    recorder.start();
    button.textContent = "Остановить";
    button.classList.add("recording");
    status.textContent = "Идёт запись";
    status.className = "error";
  };
  return { reset, toggle };
}

function selectedTestAudio(fileId, recordedBlob) {
  const file = $(fileId).files[0];
  if (file) return { blob: file, name: file.name };
  if (!recordedBlob) return null;
  const extension = recordedBlob.type.includes("ogg") ? "ogg" : recordedBlob.type.includes("wav") ? "wav" : "webm";
  return { blob: recordedBlob, name: `recording.${extension}` };
}

async function runTestStt() {
  const output = $("testSttOutput");
  const audio = selectedTestAudio("testSttFile", testState.sttBlob);
  if (!audio) throw new Error("Выберите аудиофайл или сделайте запись");
  renderTestPending(output, "Распознавание...");
  const form = new FormData();
  form.append("file", audio.blob, audio.name);
  if ($("testSttLanguage").value) form.append("language", $("testSttLanguage").value);
  const { response, milliseconds } = await testFetch("stt", { method: "POST", body: form });
  const payload = await testPayload(response);
  if (!response.ok) throw testFailure(response.status, payload);
  output.innerHTML = testMeta(response.status, milliseconds)
    + `<div class="test-answer">${escapeTest(payload.text || "Пустой результат")}</div>`
    + testRaw(payload);
}

async function runTestTts() {
  const output = $("testTtsOutput");
  renderTestPending(output, "Синтез речи...");
  const body = { text: $("testTtsText").value };
  if ($("testTtsLanguage").value) body.language = $("testTtsLanguage").value;
  if ($("testTtsBackend").value) body.backend = $("testTtsBackend").value;
  if ($("testTtsVoice").value) body.voice = $("testTtsVoice").value;
  if ($("testTtsFormat").value) body.format = $("testTtsFormat").value;
  const { response, milliseconds } = await testFetch("tts", { method: "POST", json: body });
  if (!response.ok) throw testFailure(response.status, await testPayload(response));
  const blob = await response.blob();
  if (testState.ttsUrl) URL.revokeObjectURL(testState.ttsUrl);
  testState.ttsUrl = URL.createObjectURL(blob);
  output.innerHTML = testMeta(response.status, milliseconds, `<span class="badge">${escapeTest(blob.type)}; ${(blob.size / 1024).toFixed(1)} КБ</span>`);
  const audio = document.createElement("audio");
  audio.controls = true;
  audio.autoplay = true;
  audio.src = testState.ttsUrl;
  const download = document.createElement("a");
  download.href = testState.ttsUrl;
  download.download = blob.type.includes("wav") ? "tts.wav" : blob.type.includes("mpeg") ? "tts.mp3" : "tts.bin";
  download.textContent = "Скачать аудио";
  output.append(audio, download);
}

async function runTestVoice() {
  const output = $("testVoiceOutput");
  const audio = selectedTestAudio("testVoiceFile", testState.voiceBlob);
  if (!audio) throw new Error("Выберите аудиофайл или сделайте запись");
  renderTestPending(output, "Распознавание, ответ и синтез речи...");
  const form = new FormData();
  form.append("file", audio.blob, audio.name);
  const values = {
    language: $("testVoiceLanguage").value,
    response_language: $("testVoiceResponseLanguage").value,
    tts_backend: $("testVoiceBackend").value,
    scenario_id: $("testVoiceScenario").value,
    voice: $("testVoiceVoice").value,
    subject: $("testVoiceSubject").value,
    grade: $("testVoiceGrade").value,
    lang: $("testVoiceLabLang").value,
    lab_number: $("testVoiceLabNumber").value,
  };
  for (const [name, value] of Object.entries(values)) {
    if (value) form.append(name, value);
  }
  const { response, milliseconds } = await testFetch("voice_ask", { method: "POST", body: form });
  const payload = await testPayload(response);
  if (!response.ok) throw testFailure(response.status, payload);
  const latencies = Object.entries(payload.observability?.latency_ms || {})
    .map(([name, value]) => `<span class="badge">${escapeTest(name)}: ${Math.round(value)} мс</span>`)
    .join("");
  output.innerHTML = testMeta(response.status, milliseconds, latencies)
    + `<h3>Распознанный вопрос</h3><div class="test-answer">${escapeTest(payload.question)}</div>`
    + `<h3>Ответ</h3><div class="test-answer">${escapeTest(payload.answer)}</div>`
    + testCitations(payload.citations, payload.primary_source);
  if (payload.audio_base64) {
    const playback = document.createElement("audio");
    playback.controls = true;
    playback.autoplay = true;
    playback.src = `data:${payload.audio_format || "audio/wav"};base64,${payload.audio_base64}`;
    output.append(playback);
  }
  output.insertAdjacentHTML("beforeend", testRaw({
    ...payload,
    audio_base64: payload.audio_base64 ? `${payload.audio_base64.slice(0, 40)}...` : "",
  }));
}

function bindTestButton(buttonId, outputId, action) {
  $(buttonId).addEventListener("click", async () => {
    const button = $(buttonId);
    button.disabled = true;
    try {
      await action();
    } catch (error) {
      if (error.status === 401) showLogin();
      else renderTestError($(outputId), error);
    } finally {
      button.disabled = false;
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("[data-test-view]").forEach((button) => {
    button.addEventListener("click", () => selectTestView(button.dataset.testView));
  });
  document.querySelector('[data-view="testing"]').addEventListener("click", () => {
    if (!testState.scenariosLoaded) loadTestScenarios().catch(showError);
  });
  bindTestButton("testHealthButton", "testHealthOutput", runTestHealth);
  bindTestButton("testReadyButton", "testHealthOutput", runTestReady);
  bindTestButton("testScenariosButton", "testHealthOutput", () => loadTestScenarios(true));
  bindTestButton("testAskButton", "testAskOutput", runTestAsk);
  bindTestButton("testChatButton", "testChatOutput", runTestChat);
  bindTestButton("testHintButton", "testHintOutput", runTestHint);
  bindTestButton("testSttButton", "testSttOutput", runTestStt);
  bindTestButton("testTtsButton", "testTtsOutput", runTestTts);
  bindTestButton("testVoiceButton", "testVoiceOutput", runTestVoice);
  const sttRecorder = makeTestRecorder(
    "testSttRecordButton",
    "testSttRecordStatus",
    "testSttPreview",
    "sttBlob",
    "sttPreviewUrl",
  );
  const voiceRecorder = makeTestRecorder(
    "testVoiceRecordButton",
    "testVoiceRecordStatus",
    "testVoicePreview",
    "voiceBlob",
    "voicePreviewUrl",
  );
  testState.recorderResets.push(sttRecorder.reset, voiceRecorder.reset);
  $("testSttRecordButton").addEventListener("click", sttRecorder.toggle);
  $("testVoiceRecordButton").addEventListener("click", voiceRecorder.toggle);
  for (const [inputId, labelId] of [["testSttFile", "testSttFileName"], ["testVoiceFile", "testVoiceFileName"]]) {
    $(inputId).addEventListener("change", () => {
      $(labelId).textContent = $(inputId).files[0]?.name || "Файл не выбран";
    });
  }
  selectTestView("health");
});
