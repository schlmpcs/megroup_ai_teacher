import json
import ssl
import urllib.error
import urllib.request
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response


REMOTE_BASE_URL = "https://10.9.120.4:8002"
REMOTE_TIMEOUT_SECONDS = 120


TEST_UI_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>STT/TTS Test UI</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      background: #f6f7f8;
      color: #1d252c;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      background: #f6f7f8;
    }

    main {
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 40px;
    }

    header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 20px;
      margin-bottom: 18px;
    }

    h1 {
      margin: 0;
      font-size: 28px;
      font-weight: 700;
      letter-spacing: 0;
    }

    h2 {
      margin: 0;
      font-size: 20px;
      letter-spacing: 0;
    }

    label {
      display: block;
      margin-bottom: 8px;
      font-size: 13px;
      font-weight: 650;
      color: #42515d;
    }

    textarea,
    input[type="file"] {
      width: 100%;
      border: 1px solid #cbd3d9;
      border-radius: 6px;
      background: #fff;
      color: #1d252c;
      font: inherit;
    }

    textarea {
      min-height: 126px;
      resize: vertical;
      padding: 12px;
      line-height: 1.45;
    }

    input[type="file"] {
      padding: 10px;
    }

    button {
      min-height: 38px;
      border: 1px solid #283743;
      border-radius: 6px;
      background: #283743;
      color: #fff;
      font: inherit;
      font-weight: 650;
      padding: 8px 14px;
      cursor: pointer;
    }

    button.secondary {
      background: #fff;
      color: #283743;
    }

    button:disabled {
      border-color: #b8c1c8;
      background: #d9dee2;
      color: #6b7883;
      cursor: not-allowed;
    }

    audio {
      width: 100%;
      min-height: 42px;
    }

    pre {
      min-height: 116px;
      margin: 0;
      overflow: auto;
      white-space: pre-wrap;
      border: 1px solid #d2d9de;
      border-radius: 6px;
      background: #101820;
      color: #e8edf1;
      padding: 12px;
      font-size: 13px;
      line-height: 1.45;
    }

    .status {
      align-self: stretch;
      min-width: min(460px, 100%);
      border: 1px solid #d2d9de;
      border-radius: 6px;
      background: #fff;
      padding: 12px;
      font-size: 13px;
      line-height: 1.4;
      color: #42515d;
    }

    .panels {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }

    .panel {
      display: grid;
      gap: 16px;
      border: 1px solid #d2d9de;
      border-radius: 8px;
      background: #fff;
      padding: 18px;
      box-shadow: 0 1px 2px rgba(16, 24, 32, 0.06);
    }

    .panel-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }

    .language-code {
      border-radius: 999px;
      background: #e9eef2;
      color: #344451;
      padding: 4px 9px;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }

    .control-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }

    .field {
      display: grid;
      gap: 8px;
    }

    .message {
      min-height: 20px;
      font-size: 13px;
      line-height: 1.4;
      color: #4b5a66;
    }

    .message.error {
      color: #b42318;
    }

    .message.ok {
      color: #236b35;
    }

    @media (max-width: 760px) {
      main {
        width: min(100vw - 20px, 560px);
        padding-top: 18px;
      }

      header,
      .panels {
        grid-template-columns: 1fr;
        display: grid;
      }

      h1 {
        font-size: 24px;
      }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>STT/TTS Test UI</h1>
      </div>
      <div id="health-status" class="status">Checking model status...</div>
    </header>

    <div class="panels">
      <section class="panel" data-language="kk">
        <div class="panel-title">
          <h2>Kazakh</h2>
          <span class="language-code">kk</span>
        </div>

        <div class="field">
          <label for="tts-text-kk">Text to synthesize</label>
          <textarea id="tts-text-kk" data-role="tts-text" autocomplete="off">Сәлеметсіз бе, дауыс синтезін тексеріп жатырмын.</textarea>
          <div class="control-row">
            <button data-action="tts" type="button">Generate</button>
          </div>
          <audio data-role="tts-audio" controls></audio>
          <div class="message" data-role="tts-message"></div>
        </div>

        <div class="field">
          <label for="stt-file-kk">Audio to transcribe</label>
          <input id="stt-file-kk" data-role="stt-file" type="file" accept="audio/*">
          <div class="control-row">
            <button data-action="record" type="button" class="secondary">Record</button>
            <button data-action="stop-recording" type="button" class="secondary" disabled>Stop</button>
          </div>
          <pre data-role="stt-result">Transcript will appear here.</pre>
          <div class="message" data-role="stt-message"></div>
        </div>
      </section>

      <section class="panel" data-language="ru">
        <div class="panel-title">
          <h2>Russian</h2>
          <span class="language-code">ru</span>
        </div>

        <div class="field">
          <label for="tts-text-ru">Text to synthesize</label>
          <textarea id="tts-text-ru" data-role="tts-text" autocomplete="off">Здравствуйте, я проверяю синтез речи.</textarea>
          <div class="control-row">
            <button data-action="tts" type="button">Generate</button>
          </div>
          <audio data-role="tts-audio" controls></audio>
          <div class="message" data-role="tts-message"></div>
        </div>

        <div class="field">
          <label for="stt-file-ru">Audio to transcribe</label>
          <input id="stt-file-ru" data-role="stt-file" type="file" accept="audio/*">
          <div class="control-row">
            <button data-action="record" type="button" class="secondary">Record</button>
            <button data-action="stop-recording" type="button" class="secondary" disabled>Stop</button>
          </div>
          <pre data-role="stt-result">Transcript will appear here.</pre>
          <div class="message" data-role="stt-message"></div>
        </div>
      </section>
    </div>

    <script>
      const endpoints = {
        health: "/remote/health",
        tts: "/remote/tts/synthesize",
        stt: "/remote/stt/recognize"
      };

      const recorders = new Map();

      function setMessage(panel, role, message, kind = "") {
        const element = panel.querySelector(`[data-role="${role}-message"]`);
        element.textContent = message;
        element.className = `message ${kind}`.trim();
      }

      function formatErrorPayload(payload) {
        if (typeof payload === "string") {
          return payload;
        }
        if (Array.isArray(payload)) {
          return payload.map((item) => item.msg || JSON.stringify(item)).join("; ");
        }
        if (payload && typeof payload === "object") {
          return payload.detail ? formatErrorPayload(payload.detail) : JSON.stringify(payload);
        }
        return "Request failed";
      }

      async function readError(response) {
        const contentType = response.headers.get("content-type") || "";
        if (contentType.includes("application/json")) {
          return formatErrorPayload(await response.json());
        }
        return await response.text();
      }

      async function loadHealth() {
        const status = document.querySelector("#health-status");
        try {
          const response = await fetch(endpoints.health);
          if (!response.ok) {
            throw new Error(await readError(response));
          }
          const health = await response.json();
          status.textContent = `Status: ${health.status} | STT: ${health.stt_models.join(", ") || "none"} | TTS: ${health.tts_models.join(", ") || "none"}`;
        } catch (error) {
          status.textContent = `Health check failed: ${error.message}`;
        }
      }

      async function synthesize(panel) {
        const language = panel.dataset.language;
        const text = panel.querySelector('[data-role="tts-text"]').value.trim();
        const audio = panel.querySelector('[data-role="tts-audio"]');
        if (!text) {
          setMessage(panel, "tts", "Enter text before generating audio.", "error");
          return;
        }

        setMessage(panel, "tts", "Generating audio...");
        try {
          const response = await fetch(endpoints.tts, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text, language, speed: 1.0 })
          });
          if (!response.ok) {
            throw new Error(await readError(response));
          }
          const blob = await response.blob();
          if (audio.dataset.objectUrl) {
            URL.revokeObjectURL(audio.dataset.objectUrl);
          }
          const objectUrl = URL.createObjectURL(blob);
          audio.dataset.objectUrl = objectUrl;
          audio.src = objectUrl;
          setMessage(panel, "tts", "Audio generated.", "ok");
        } catch (error) {
          setMessage(panel, "tts", error.message, "error");
        }
      }

      async function transcribe(panel, file) {
        if (!file) {
          return;
        }
        const result = panel.querySelector('[data-role="stt-result"]');
        const body = new FormData();
        body.append("language", panel.dataset.language);
        body.append("audio", file, file.name || `recording-${panel.dataset.language}.webm`);

        result.textContent = "Transcribing...";
        setMessage(panel, "stt", "");
        try {
          const response = await fetch(endpoints.stt, {
            method: "POST",
            body
          });
          if (!response.ok) {
            throw new Error(await readError(response));
          }
          const payload = await response.json();
          result.textContent = [
            `Text: ${payload.text || ""}`,
            `Language: ${payload.language || ""}`,
            `Duration: ${payload.duration_ms ?? ""} ms`,
            `Confidence: ${payload.confidence ?? ""}`
          ].join("\\n");
          setMessage(panel, "stt", "Transcript ready.", "ok");
        } catch (error) {
          result.textContent = "Transcript will appear here.";
          setMessage(panel, "stt", error.message, "error");
        }
      }

      async function startRecording(panel) {
        if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
          setMessage(panel, "stt", "Browser recording is not supported here. Use audio upload.", "error");
          return;
        }

        const recordButton = panel.querySelector('[data-action="record"]');
        const stopButton = panel.querySelector('[data-action="stop-recording"]');
        const chunks = [];

        try {
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          const recorder = new MediaRecorder(stream);
          recorders.set(panel, { recorder, stream });

          recorder.addEventListener("dataavailable", (event) => {
            if (event.data.size > 0) {
              chunks.push(event.data);
            }
          });

          recorder.addEventListener("stop", () => {
            stream.getTracks().forEach((track) => track.stop());
            recorders.delete(panel);
            recordButton.disabled = false;
            stopButton.disabled = true;
            const blob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
            const file = new File([blob], `recording-${panel.dataset.language}.webm`, { type: blob.type });
            transcribe(panel, file);
          });

          recorder.start();
          recordButton.disabled = true;
          stopButton.disabled = false;
          setMessage(panel, "stt", "Recording...");
        } catch (error) {
          setMessage(panel, "stt", error.message, "error");
        }
      }

      function stopRecording(panel) {
        const active = recorders.get(panel);
        if (active?.recorder?.state === "recording") {
          active.recorder.stop();
        }
      }

      document.querySelectorAll(".panel").forEach((panel) => {
        panel.querySelector('[data-action="tts"]').addEventListener("click", () => synthesize(panel));
        panel.querySelector('[data-role="stt-file"]').addEventListener("change", (event) => transcribe(panel, event.target.files[0]));
        panel.querySelector('[data-action="record"]').addEventListener("click", () => startRecording(panel));
        panel.querySelector('[data-action="stop-recording"]').addEventListener("click", () => stopRecording(panel));
      });

      loadHealth();
    </script>
  </main>
</body>
</html>
"""


def register_ui(app: FastAPI) -> None:
    @app.get("/ui", response_class=HTMLResponse)
    def ui() -> HTMLResponse:
        return HTMLResponse(TEST_UI_HTML)


def _remote_url(path: str) -> str:
    return f"{REMOTE_BASE_URL}{path}"


def _unverified_tls_context() -> ssl.SSLContext:
    return ssl._create_unverified_context()


def _proxy_request(request: urllib.request.Request) -> Response:
    try:
        with urllib.request.urlopen(
            request,
            timeout=REMOTE_TIMEOUT_SECONDS,
            context=_unverified_tls_context(),
        ) as remote_response:
            content = remote_response.read()
            headers = getattr(remote_response, "headers", None)
            content_type = headers.get("Content-Type", "application/octet-stream") if headers else "application/octet-stream"
            return Response(content=content, media_type=content_type)
    except urllib.error.HTTPError as exc:
        detail = exc.read()
        content_type = exc.headers.get("Content-Type", "application/json") if exc.headers else "application/json"
        return Response(content=detail, status_code=exc.code, media_type=content_type)
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=502, detail=f"Remote request failed: {exc.reason}") from exc


def _multipart_body(language: str, filename: str, content_type: str, audio_bytes: bytes) -> tuple[bytes, str]:
    boundary = f"----vrrag-ui-{uuid.uuid4().hex}"
    parts = [
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="language"\r\n\r\n'
        f"{language}\r\n",
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="audio"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n",
    ]
    body = b"".join(part.encode("utf-8") for part in parts)
    body += audio_bytes
    body += f"\r\n--{boundary}--\r\n".encode("utf-8")
    return body, boundary


def register_remote_proxy(app: FastAPI) -> None:
    @app.get("/remote/health")
    def remote_health() -> Response:
        request = urllib.request.Request(_remote_url("/health"), method="GET")
        return _proxy_request(request)

    @app.post("/remote/tts/synthesize")
    async def remote_tts(payload: dict) -> Response:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            _remote_url("/tts/synthesize"),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return _proxy_request(request)

    @app.post("/remote/stt/recognize")
    async def remote_stt(audio: UploadFile = File(...), language: str = Form(default="auto")) -> Response:
        audio_bytes = await audio.read()
        body, boundary = _multipart_body(
            language=language,
            filename=audio.filename or "audio.wav",
            content_type=audio.content_type or "application/octet-stream",
            audio_bytes=audio_bytes,
        )
        request = urllib.request.Request(
            _remote_url("/stt/recognize"),
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        return _proxy_request(request)


def create_ui_app() -> FastAPI:
    ui_app = FastAPI(title="VRRAG Remote STT/TTS Test UI")
    register_ui(ui_app)
    register_remote_proxy(ui_app)

    @ui_app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/ui")

    return ui_app


app = create_ui_app()
