# Task 3 Report

Date: July 22, 2026
Branch: `feature/admin-ingestion-ui`
Commit target: `feat: add safe ingestion progress primitives`

## Scope

Implemented Task 3 in `app/services/ingestion.py` with focused test coverage in
`tests/test_ingestion_safety_units.py`.

Preserved:
- generation-based replacement semantics in `upload_document()`
- duplicate-lab detection during bulk ingest
- `bulk_ingest_tree(root, ocr=None, only=None, prune=False)` CLI semantics
- unrecognized-path skips as non-fatal
- OCR default handling
- page and section locator behavior
- prune guard rejecting `prune=True` with `only=...`

## RED

Added the requested failing tests first:
- `test_resolve_corpus_scope_rejects_traversal_and_escaping_symlink`
- `test_scan_corpus_tree_reports_same_candidates_as_bulk_validation`
- `test_scan_corpus_tree_reports_duplicate_lab_ids`
- `test_upload_document_reports_stages_and_stops_before_indexing`

Command:

```bash
pytest tests/test_ingestion_safety_units.py tests/test_retrieval_units.py -q
```

Result:
- failed as expected
- 4 failures, 47 passes
- missing interfaces:
  - `ingestion.resolve_corpus_scope`
  - `ingestion.scan_corpus_tree`
  - `ingestion.IngestionCancelled`

## Implementation

Changed `app/services/ingestion.py` to add:
- `ProgressCallback` and `CancelCheck` callback types
- `IngestionCancelled`
- `_stage()` safe-boundary helper
- `resolve_corpus_scope()`
- `scan_corpus_tree()`
- `prune_missing_corpus_documents()`
- optional `progress` and `should_cancel` hooks on `upload_document()`

Behavior details:
- cancellation is checked before each safe stage boundary
- stage callbacks fire at `extracting`, `embedding`, and `indexing`
- corpus subtree resolution now rejects traversal and escaping symlinks
- bulk ingest now reuses a single scan pass for candidate selection and prune
  inputs
- prune still runs only for explicit full snapshots with candidates and no
  errors
- unrecognized corpus paths still count as skipped without becoming fatal

## GREEN

Focused commands:

```bash
pytest tests/test_ingestion_safety_units.py tests/test_retrieval_units.py tests/test_manage_corpus_cli_units.py -q
```

Result:
- 57 passed

Full suite:

```bash
pytest -q
```

Result:
- 689 passed

## Files Changed

- `app/services/ingestion.py`
- `tests/test_ingestion_safety_units.py`

## Self-review

- Kept the diff inside the existing ingestion module and safety tests.
- Reused existing metadata validation and duplicate-lab logic instead of adding
  a second path.
- Left manifest behavior untouched.
- Left retrieval behavior untouched except for preserving the existing tests.
- Verified cancellation stops before any indexing call in the new stage test.

## Concerns

- `scan_corpus_tree()` returns absolute filesystem paths for candidate items.
  That matches the current internal use, but it is an internal helper and
  should stay server-side.
- Focused and full pytest were clean; no open functional issue found in this
  task scope.
