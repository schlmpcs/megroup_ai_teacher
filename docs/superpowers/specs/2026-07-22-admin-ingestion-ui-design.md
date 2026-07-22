# Admin Ingestion UI Design

**Date:** 2026-07-22

## Summary

Build a standalone admin web service for managing knowledge-base ingestion. The
browser authenticates to the admin service with one administrator account. The
admin service keeps the backend URL and admin API key server-side, so neither is
exposed to browser JavaScript.

Long-running ingestion moves to a persistent SQLite queue consumed by one
dedicated worker process. This keeps OCR, parsing, embedding, and Qdrant writes
out of the latency-sensitive VR API process while supporting durable progress,
retry history, browser batches, and server-side corpus ingestion.

## Existing Capabilities

The backend already provides the core synchronous operations:

- `POST /admin/documents` for one general or structured document.
- `GET /admin/documents` for document listing.
- `DELETE /admin/documents/{file_id}` for deletion.
- `GET /admin/corpus_status` for Qdrant collection status.
- `ingestion.upload_document(...)` for extraction, chunking, embedding, and
  Qdrant replacement semantics.
- `ingestion.bulk_ingest_tree(...)` for path-derived corpus ingestion.
- `corpus_meta.build_upload_metadata(...)` and `corpus_meta.parse_path(...)`
  for canonical metadata and document identity.

`test_ui.html` contains a developer upload form, but it is not an authenticated,
deployed, or operational admin application. The new UI is a separate service.

## Goals

1. Provide one administrator login without exposing backend credentials.
2. Upload one file, multiple files, or a browser-selected folder.
3. Auto-detect metadata and allow per-file edits before queueing.
4. Preview and queue ingestion from the backend server's configured
   `CORPUS_ROOT` or a safe relative subtree.
5. Persist jobs, item progress, errors, and retry history across restarts.
6. Run exactly one ingestion job at a time in FIFO order.
7. List and delete ingested documents through the existing backend behavior.
8. Keep VR request handling responsive while ingestion runs.

## Non-Goals

- Multiple administrator accounts, roles, or SSO.
- Concurrent ingestion workers.
- Redis, Celery, or an external database server.
- Arbitrary server filesystem browsing.
- Scheduled ingestion.
- WebSocket or SSE job updates.
- Automatic job artifact expiry.
- Editing document contents after upload.
- Antivirus or content moderation for trusted internal administrator uploads.

## Architecture

### Services

The deployed system gains two services:

1. `admin-ui`
   - Separate Python FastAPI service with static HTML, CSS, and JavaScript.
   - Handles login, signed sessions, CSRF checks, and explicit backend proxy
     routes.
   - Stores backend connection details only in server environment variables.
   - Can be built and deployed independently from the backend image.

2. `ingestion-worker`
   - Uses the same image and source revision as the existing `api` service.
   - Has no public port.
   - Polls SQLite, claims the oldest queued job, and runs existing ingestion
     functions.
   - Mounts `CORPUS_ROOT` read-only and the ingestion data volume read-write.

The existing `api` service owns job creation and job-query endpoints. It streams
browser uploads into persistent storage and writes job records, but it does not
run document extraction or embedding. Both `api` and `ingestion-worker` mount
`CORPUS_ROOT` read-only because the API performs metadata-only corpus previews
while the worker performs ingestion.

The existing synchronous `POST /admin/documents` endpoint remains for backward
compatibility. The new admin UI never uses it; all UI ingestion goes through the
job queue.

### Request Path

```text
browser
  -> admin-ui session authentication + CSRF
  -> admin-ui explicit proxy route
  -> backend /admin/* with ADMIN_API_KEY
  -> SQLite job row + persistent upload files
  -> ingestion-worker claims job
  -> existing ingestion service
  -> embedder + Qdrant
  -> worker updates SQLite progress
  -> admin-ui polls backend job status
```

The browser never receives `ADMIN_API_KEY`, `INTERNAL_API_KEY`, or the internal
backend URL.

## Authentication And Authorization

### Backend Separation

Add required backend setting `ADMIN_API_KEY`. Every `/admin/*` route uses a new
admin-key dependency. `INTERNAL_API_KEY` continues to authorize VR-facing routes
but no longer authorizes admin routes.

Existing command-line ingestion scripts call service functions directly and do
not require the HTTP admin key.

### Admin UI Login

The admin service uses these settings:

- `ADMIN_UI_USERNAME`
- `ADMIN_UI_PASSWORD_HASH`
- `ADMIN_UI_SESSION_SECRET`
- `ADMIN_UI_SESSION_TTL_S=28800`
- `ADMIN_UI_COOKIE_SECURE=true`
- `BACKEND_BASE_URL=http://api:8000`
- `BACKEND_ADMIN_API_KEY`

`ADMIN_UI_PASSWORD_HASH` uses `hashlib.scrypt` with `n=16384`, `r=8`, `p=1`, and
a 32-byte derived key. Password comparison uses `hmac.compare_digest`.

Successful login creates an HMAC-SHA256 signed session containing the username,
issue time, expiry time, and a random CSRF token. The cookie is named
`admin_session` and is set with `HttpOnly`, `SameSite=Strict`, `Path=/`, and the
configured `Secure` flag. Logout expires the cookie.

Login is limited to five failed attempts per minute per client IP. Error messages
do not reveal whether the username or password was incorrect.

Every mutating same-origin request requires `X-CSRF-Token` matching the token in
the signed session. The admin service has no permissive CORS configuration.

### Proxy Restrictions

The admin service exposes explicit handlers only for the required backend
operations. It is not an arbitrary URL or path proxy. It injects
`Authorization: Bearer <BACKEND_ADMIN_API_KEY>` server-side and removes incoming
authorization headers from the browser request.

Authorization headers, passwords, session cookies, CSRF tokens, and backend keys
must not be logged.

## Persistent Job Storage

### Filesystem

The backend and worker share one named volume mounted at `/data/ingestion`:

```text
/data/ingestion/
  jobs.sqlite3
  tmp/<job_id>/
  uploads/<job_id>/<item_id>
```

Temporary upload directories are removed after an interrupted or rejected
request. A valid batch is atomically renamed from `tmp/<job_id>` to
`uploads/<job_id>` before its job becomes visible as `queued`.

Stored upload filenames are generated item IDs. Original filenames and browser
relative paths remain metadata only and are never used as filesystem paths. On
API startup, temporary directories older than 24 hours with no matching job are
removed.

Uploaded source files remain until the administrator deletes the corresponding
terminal job. Retry jobs copy only the items being retried into their own job
directory, preserving independent deletion semantics. Server-corpus jobs store
parameters only and rescan the filesystem on retry.

### SQLite

Use Python's standard `sqlite3` module with:

- `PRAGMA journal_mode=WAL`
- `PRAGMA foreign_keys=ON`
- `PRAGMA busy_timeout=5000`
- `PRAGMA user_version=1`

Each operation opens a short-lived connection. No ORM or migration framework is
required for schema version 1.

`jobs` stores:

- UUID job ID.
- Kind: `upload` or `corpus`.
- Status: `queued`, `running`, `completed`, `partial`, `failed`, or `cancelled`.
- JSON options.
- Optional `retry_of` job ID.
- Cancellation flag.
- Total, completed, failed, and skipped item counts.
- Current filename and stage.
- Sanitized terminal error or warning.
- UTC created, started, and finished timestamps.

`job_items` stores:

- UUID item ID and owning job ID.
- Stable position in the batch.
- Filename and optional browser-relative path.
- Stored file path for browser uploads.
- Validated metadata JSON and OCR flag.
- Status: `pending`, `running`, `completed`, `failed`, `skipped`, or `cancelled`.
- Stage: `queued`, `extracting`, `embedding`, `indexing`, or `done`.
- Resulting document ID, chunk count, and sanitized error.
- UTC started and finished timestamps.

The worker also writes a heartbeat row every five seconds. A heartbeat older
than fifteen seconds is reported as offline by the job-status endpoint.

## Job Claiming And Recovery

The worker polls once per second. Claiming uses `BEGIN IMMEDIATE`, selects the
oldest `queued` job by creation time, changes it to `running`, and commits before
work begins. This remains correct if an additional worker is started by mistake,
although deployment supports only one worker.

On startup, the worker marks any previously `running` job as `failed` with an
interruption reason. Its completed item results remain visible. Any `running` or
`pending` items from that job become `failed` with the same reason, making the
retry set unambiguous. The administrator can retry those failed items as a new
linked job.

Cancellation behavior:

- A queued job becomes `cancelled` immediately.
- A running job sets `cancel_requested=1`.
- The worker checks cancellation before each file and between extraction,
  embedding, and indexing where the existing ingestion flow has a safe boundary.
- An active OCR call or Qdrant write is allowed to finish before cancellation.
- Remaining pending items become `cancelled`.

## Browser Upload Flow

### Staging

The browser supports both a normal multiple-file input and a folder input using
the native `webkitdirectory` capability where available. Files appear in a
staging table before upload.

Each row contains:

- Filename and relative path.
- Document kind shown as General, Textbook, or Lab instruction.
- Subject: physics, chemistry, or biology.
- Grade: 7 through 11.
- Language: ru, kk, or en.
- Lab number: 1 through 99 when required.
- OCR toggle.
- Validation status.

Folder paths are sent to the backend preview endpoint. The backend reuses
`corpus_meta.parse_path(...)` to suggest metadata. The administrator can edit any
suggestion. General documents submit all structured metadata fields as null.

The backend preview and final submission both detect duplicate deterministic
document identities inside one batch. A batch cannot queue two general documents
with the same identity, two replacements for the same textbook identity, or two
authoritative instructions for the same `lab_id`.

### Submission

`POST /admin/ingestion/jobs/upload` accepts multipart form data:

- Repeated `files` fields in staging-table order.
- One `manifest` JSON field containing an entry for each file in the same order.

Each manifest entry contains:

```json
{
  "filename": "Lab work 2.docx",
  "relative_path": "Laboratory works/Physics/Physics Grade 10/en/Lab work 2.docx",
  "doc_type": "lab_instruction",
  "subject": "physics",
  "grade": 10,
  "lang": "en",
  "lab_number": 2,
  "ocr": false
}
```

The backend validates manifest length, filename order, safe basenames,
extensions, sizes, metadata combinations, and duplicate identities before
creating the job. Structured identity is built with the existing
`build_upload_metadata(...)` behavior. Relative paths provide suggestions only;
browser uploads remain under the existing `admin_uploads` identity namespace.

Limits:

- Existing `MAX_DOCUMENT_UPLOAD_BYTES` applies to each file.
- `INGESTION_BATCH_MAX_FILES=100` limits one browser job.
- `INGESTION_BATCH_MAX_BYTES=1073741824` limits one browser job to 1 GiB.

The endpoint returns HTTP `202` with the created job summary.

## Server Corpus Flow

### Path Confinement

Server-corpus requests accept only an optional relative `subtree`. The backend
resolves both `CORPUS_ROOT` and the candidate path, requires the candidate to be
beneath the root, and rejects absolute paths, `..` traversal, and symlinks that
escape the root.

### Preview

`POST /admin/ingestion/corpus/preview` accepts:

```json
{
  "subtree": "School materials/Biology/en",
  "ocr": false,
  "prune": false
}
```

It returns recognized files, skipped files with reasons, duplicate lab IDs,
document counts by language/type, and the number of existing corpus-owned
documents that a full-root prune would remove.

Preview and execution use the same corpus-scan helper so their metadata and
safety decisions cannot drift.

### Queueing

`POST /admin/ingestion/jobs/corpus` accepts the same JSON and returns HTTP `202`.
The worker rescans at execution time, writes item rows, and calls the existing
bulk-ingestion behavior with progress callbacks.

`prune=true` is allowed only when `subtree` is empty. It requires explicit UI
confirmation and can delete only corpus-owned documents. It must never delete
documents whose source path begins with `admin_uploads/`.

## Backend API

New protected endpoints:

- `GET /admin/ingestion/status`
- `POST /admin/ingestion/preview`
- `POST /admin/ingestion/corpus/preview`
- `POST /admin/ingestion/jobs/upload`
- `POST /admin/ingestion/jobs/corpus`
- `GET /admin/ingestion/jobs`
- `GET /admin/ingestion/jobs/{job_id}`
- `POST /admin/ingestion/jobs/{job_id}/cancel`
- `POST /admin/ingestion/jobs/{job_id}/retry`
- `DELETE /admin/ingestion/jobs/{job_id}`
- `POST /admin/cache/answers/clear`

Existing protected endpoints remain:

- `GET /admin/corpus_status`
- `POST /admin/documents`
- `GET /admin/documents`
- `DELETE /admin/documents/{file_id}`
- `GET /admin/scenarios`

Job list responses support `status`, `kind`, `limit`, and `offset` filters.
Default `limit` is 50 and maximum `limit` is 200. All timestamps are ISO 8601
UTC strings.

Retry creates a new job with `retry_of` set to the original job ID. Upload-job
retry includes failed and cancelled items but excludes skipped items and items
that already completed successfully. Corpus retry repeats the original scan
options and rescans the current filesystem.

Job deletion is allowed only for terminal jobs. It removes the job rows and its
own retained upload directory. Active jobs must be cancelled and reach a
terminal state first.

## Cache Invalidation

Direct document upload and deletion continue to clear the in-process answer
cache in the API process.

The ingestion worker is a separate process, so after any successful upload,
replacement, deletion, or prune it calls `POST /admin/cache/answers/clear` over
the internal network with `ADMIN_API_KEY`. The worker retries this callback three
times. If corpus mutation succeeded but cache invalidation still fails, the job
finishes as `partial` with a visible warning rather than reporting ingestion as
failed. This callback still runs when a cancelled or failed job mutated at least
one document before reaching its terminal state.

## Admin UI

### Screens

1. Login
   - Username and password fields.
   - Generic invalid-credentials error.

2. Ingest, the default authenticated screen
   - Status strip for backend, Qdrant, worker heartbeat, queue count, and active
     job.
   - Segmented source control: Browser upload or Server corpus.
   - Browser staging table with editable metadata and validation.
   - Server preview with OCR and guarded prune controls.

3. Jobs
   - Active FIFO queue followed by persistent history.
   - Status, kind, timestamps, progress, current file/stage, and result counts.
   - Expandable per-file errors.
   - Cancel, retry, and delete actions where valid.

4. Documents
   - Corpus summary.
   - Filterable table for filename, type, subject, grade, and language.
   - Confirmation before document deletion.

### Interaction Rules

- English operator copy only for version 1.
- Active jobs poll every two seconds. Idle screens poll every ten seconds.
- Queue buttons remain disabled while any staging row is invalid.
- Destructive actions require confirmation.
- Forms have explicit labels, keyboard focus styles, and screen-reader status
  announcements.
- Tables use stable column widths and become horizontally scrollable on narrow
  screens rather than collapsing content into nested cards.
- The UI stores no API keys or passwords in local storage or session storage.

## Error Handling

- Upload request failure before queue creation removes temporary files and
  creates no job.
- Per-file worker failures are recorded and processing continues with the next
  file.
- A non-cancelled job with at least one success and at least one failure becomes
  `partial`.
- A non-cancelled job with at least one success and no failures becomes
  `completed`; skipped corpus files remain visible in its counts.
- A job with no successful items and any failure, or with no ingestible items,
  becomes `failed`.
- A user-cancelled job becomes `cancelled` even if earlier items succeeded; its
  successful item results remain visible.
- Unsupported or unrecognized corpus files become `skipped` with a reason.
- Backend unavailability is shown as a service error without logging secrets.
- Qdrant and embedder errors retain the existing service error mapping and are
  copied into sanitized per-file job errors.
- Document deletion keeps the existing 404 behavior for missing IDs.

## Deployment

Add to Docker Compose:

- `admin-ui`, exposed on host port 8004 by default.
- `ingestion-worker`, no exposed port, `restart: unless-stopped`.
- `ingestion_data` named volume shared by `api` and `ingestion-worker`.
- Read-only `CORPUS_ROOT` mount on `api` and `ingestion-worker`.

The `api` and `ingestion-worker` services must run the same image tag so their
SQLite schema and ingestion code match. `admin-ui` has its own Dockerfile and
small requirements file, so it can be deployed independently.

TLS remains the responsibility of the existing reverse proxy or private network
edge. Production sets `ADMIN_UI_COOKIE_SECURE=true`.

## Testing

### Backend And Worker

- Admin routes reject `INTERNAL_API_KEY` and accept `ADMIN_API_KEY`.
- SQLite schema creation, FIFO claim, status transitions, and pagination.
- Worker heartbeat and stale-worker reporting.
- Startup recovery of abandoned running jobs.
- Atomic browser upload staging and cleanup on failure.
- Per-file and per-batch limits.
- Metadata validation and duplicate identity rejection.
- Corpus path traversal and escaping-symlink rejection.
- Preview/execution scan consistency.
- Partial batches, cancellation, retries, and terminal deletion rules.
- Full-root prune protection for `admin_uploads` documents.
- Cache invalidation success and failure warning behavior.

All Qdrant, embedder, and external HTTP calls remain mocked in unit tests.

### Admin UI

- Login success/failure, rate limit, session expiry, logout, and cookie flags.
- CSRF rejection for missing or invalid tokens.
- Backend authorization injection and browser authorization stripping.
- Proxy path allowlist.
- Secret non-disclosure in HTML, JSON, logs, and browser storage.
- Playwright flows for login, staging, metadata editing, queueing, progress,
  retry, corpus preview, document filtering, and deletion.

### Verification

- Run the complete existing `pytest` suite.
- Run the new admin service and worker tests without network or GPU access.
- Validate Docker Compose configuration.
- On the deployed GPU server, queue one small browser upload and one restricted
  corpus subtree job.
- While a job runs, verify `/ask` remains responsive from the existing API
  service.
- Restart the worker during a test job, verify the job becomes failed, and retry
  it successfully.

## Acceptance Criteria

1. An unauthenticated browser cannot access admin data or mutation endpoints.
2. The backend admin key never appears in browser responses, browser storage, or
   logs.
3. A valid browser batch can be staged, edited, queued, monitored, retried, and
   deleted from the UI.
4. A server-corpus job cannot escape `CORPUS_ROOT`.
5. Job history and uploaded retry files survive API, worker, and admin UI
   restarts.
6. At most one ingestion job is `running`.
7. Partial failures preserve successful documents and show per-file errors.
8. Successful corpus mutations invalidate the API answer cache or produce a
   visible warning if invalidation fails.
9. Existing direct document listing and deletion behavior remains compatible,
   except that all admin endpoints require `ADMIN_API_KEY`.
10. OCR and embedding work run outside the VR API process.
