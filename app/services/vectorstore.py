"""Local hybrid-RAG vector store backed by Qdrant.

The knowledge base lives in a single Qdrant collection that stores two named
vectors per chunk:

- ``"dense"``  — a normalised bge-m3 dense embedding (cosine distance).
- ``"sparse"`` — bge-m3 learned sparse weights (no IDF modifier; the model
  already produces calibrated term weights).

At query time we prefetch candidates from each branch independently and fuse
them with Reciprocal Rank Fusion (RRF) via the Qdrant Query API. This module
is the only place that talks to Qdrant; everything else goes through the
``async`` functions below.

Failures are mapped onto the shared service-layer exceptions
(:class:`LLMTimeoutError` / :class:`LLMUpstreamError`) so routes can translate
them to 504/502 just like OpenAI failures.
"""

import asyncio
import logging
import uuid

from qdrant_client import AsyncQdrantClient, models

from app.core.config import settings
from app.core.languages import SUPPORTED_LANGUAGES
from app.services.errors import LLMError, LLMTimeoutError, LLMUpstreamError

logger = logging.getLogger("assistant.vectorstore")

DENSE = "dense"
SPARSE = "sparse"

# Page size for full-collection scans (the school KB is small).
_SCROLL_PAGE = 256

# Lazy, patchable singleton client (tests monkeypatch ``_client`` or ``get_client``).
_client: AsyncQdrantClient | None = None

# ponytail: one write lock is enough for admin/bulk ingest; use per-document
# locks if concurrent ingest throughput becomes important.
_upsert_lock = asyncio.Lock()
_collection_lock = asyncio.Lock()


def _point_id_key(point_id) -> str:
    try:
        return uuid.UUID(str(point_id)).hex
    except ValueError:
        return str(point_id)


def _require_completed(result, operation: str) -> None:
    status = getattr(result, "status", None)
    if status == models.UpdateStatus.COMPLETED:
        return
    value = getattr(status, "value", status)
    error = (
        LLMTimeoutError
        if status == models.UpdateStatus.WAIT_TIMEOUT
        else LLMUpstreamError
    )
    raise error(f"Qdrant {operation} status: {value}")


def get_client() -> AsyncQdrantClient:
    """Return the shared Qdrant client, creating it on first use."""
    global _client
    if _client is None:
        _client = AsyncQdrantClient(url=settings.QDRANT_URL)
    return _client


def _map_qdrant_error(exc: Exception) -> LLMError:
    """Translate a Qdrant/transport failure into a service-layer exception."""
    name = type(exc).__name__
    if "Timeout" in name or "Connection" in name or name == "ResponseHandlingException":
        return LLMTimeoutError(f"Qdrant unreachable: {exc}")
    return LLMUpstreamError(f"Qdrant error: {exc}")


def _doc_filter(doc_id: str) -> models.Filter:
    """Filter matching all points belonging to ``doc_id``."""
    return models.Filter(
        must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
    )


def _resolve_collection_name(collection_name: str | None) -> str:
    return settings.QDRANT_COLLECTION if collection_name is None else collection_name


def meta_filter(**fields) -> models.Filter | None:
    """Build a Qdrant ``must``-filter from non-None payload fields.

    e.g. ``meta_filter(doc_type="textbook", subject="physics")`` scopes a search
    to physics theory. Returns None when no constraints were given so callers can
    pass it straight through to :func:`hybrid_search`.
    """
    must = [
        models.FieldCondition(key=key, match=models.MatchValue(value=value))
        for key, value in fields.items()
        if value is not None
    ]
    return models.Filter(must=must) if must else None


def with_lang(base: "models.Filter | None", lang: str | None) -> "models.Filter | None":
    """Return ``base`` narrowed to a single ``lang`` (e.g. "ru"/"kk").

    Adds a ``lang`` ``must``-condition on top of any conditions already in
    ``base`` (subject/doc_type scope), leaving ``base`` untouched. When ``lang``
    is None the base filter is returned as-is, so callers can pass it straight
    through to :func:`hybrid_search`. Used to prefer same-language chunks, with
    the unconstrained ``base`` kept as a fallback.
    """
    if not lang:
        return base
    lang_cond = models.FieldCondition(key="lang", match=models.MatchValue(value=lang))
    base_must = list(getattr(base, "must", None) or []) if base is not None else []
    return models.Filter(must=base_must + [lang_cond])


def _without_inactive(base: "models.Filter | None") -> models.Filter:
    condition = models.FieldCondition(
        key="status",
        match=models.MatchAny(any=["pending", "staging", "superseded"]),
    )
    if base is None:
        return models.Filter(must_not=[condition])
    return base.model_copy(
        update={"must_not": [*(base.must_not or []), condition]}
    )


def _with_generation_exclusions(
    base: models.Filter,
    generations: set[str],
    legacy_doc_ids: set[str],
    point_ids: list,
) -> models.Filter:
    must_not = list(base.must_not or [])
    if generations:
        must_not.append(
            models.FieldCondition(
                key="generation",
                match=models.MatchAny(any=sorted(generations)),
            )
        )
    if legacy_doc_ids:
        must_not.append(
            models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchAny(any=sorted(legacy_doc_ids)),
                    ),
                    models.IsEmptyCondition(
                        is_empty=models.PayloadField(key="generation")
                    ),
                ]
            )
        )
    if point_ids:
        must_not.append(models.HasIdCondition(has_id=point_ids))
    return base.model_copy(update={"must_not": must_not})


async def ensure_collection(collection_name: str | None = None) -> None:
    """Create the dense+sparse collection if it does not already exist."""
    collection = _resolve_collection_name(collection_name)
    try:
        async with _collection_lock:
            client = get_client()
            if await client.collection_exists(collection):
                return
            try:
                await client.create_collection(
                    collection_name=collection,
                    vectors_config={
                        DENSE: models.VectorParams(
                            size=settings.EMBEDDING_DIM,
                            distance=models.Distance.COSINE,
                        )
                    },
                    sparse_vectors_config={SPARSE: models.SparseVectorParams()},
                )
            except Exception:  # noqa: BLE001 - another process may have won
                try:
                    exists = await client.collection_exists(collection)
                except Exception:  # noqa: BLE001 - preserve the create failure
                    exists = False
                if not exists:
                    raise
                return
            logger.info("Created Qdrant collection '%s'", collection)
    except Exception as exc:  # noqa: BLE001 - normalise to service-layer errors
        raise _map_qdrant_error(exc) from exc


async def upsert_points(
    points: list[dict], collection_name: str | None = None
) -> int:
    """Upsert hybrid points and return how many were written.

    Each point dict carries ``id``, a ``dense`` vector, ``sparse_indices`` /
    ``sparse_values`` for the sparse vector, and a ``payload``. Empty input is
    a no-op (no network call) and returns 0.
    """
    if not points:
        return 0
    collection = _resolve_collection_name(collection_name)
    doc_ids = {point["payload"].get("doc_id") for point in points}
    replace_doc_id = doc_ids.pop() if len(doc_ids) == 1 else None
    generations = {
        _payload_generation(point["payload"])
        for point in points
        if _payload_generation(point["payload"]) is not None
    }
    generation = generations.pop() if len(generations) == 1 else None
    activate_generation = generation if replace_doc_id and generation and all(
        _payload_generation(point["payload"]) == generation for point in points
    ) else None
    structs = [
        models.PointStruct(
            id=p["id"],
            vector={
                DENSE: p["dense"],
                SPARSE: models.SparseVector(
                    indices=p["sparse_indices"],
                    values=p["sparse_values"],
                ),
            },
            payload={
                **p["payload"],
                **({"status": "staging"} if activate_generation else {}),
            },
        )
        for p in points
    ]
    new_ids = {_point_id_key(point["id"]) for point in points}
    client = get_client()
    try:
        async with _upsert_lock:
            old_ids: dict[str, object] = {}
            if replace_doc_id:
                offset = None
                while True:
                    records, offset = await client.scroll(
                        collection_name=collection,
                        scroll_filter=_doc_filter(replace_doc_id),
                        limit=_SCROLL_PAGE,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False,
                    )
                    for record in records:
                        if _is_visible_payload(
                            dict(getattr(record, "payload", None) or {})
                        ):
                            old_ids[_point_id_key(record.id)] = record.id
                    if offset is None:
                        break

            upsert_result = await client.upsert(
                collection_name=collection,
                points=structs,
            )
            _require_completed(upsert_result, "upsert")

            if activate_generation and replace_doc_id:
                activation_result = await client.set_payload(
                    collection_name=collection,
                    payload={
                        "active_generation": activate_generation,
                        "status": "ready",
                    },
                    points=_doc_filter(replace_doc_id),
                )
                _require_completed(activation_result, "generation activation")

                supersede_result = await client.set_payload(
                    collection_name=collection,
                    payload={"status": "superseded"},
                    points=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="doc_id",
                                match=models.MatchValue(value=replace_doc_id),
                            ),
                            models.FieldCondition(
                                key="active_generation",
                                match=models.MatchValue(value=activate_generation),
                            ),
                        ],
                        must_not=[
                            models.FieldCondition(
                                key="generation",
                                match=models.MatchValue(value=activate_generation),
                            )
                        ],
                    ),
                )
                _require_completed(supersede_result, "generation supersede")

            stale_ids = [
                point_id for key, point_id in old_ids.items() if key not in new_ids
            ]
            if stale_ids:
                try:
                    selector = (
                        models.FilterSelector(
                            filter=models.Filter(
                                must=[
                                    models.HasIdCondition(has_id=stale_ids),
                                    models.FieldCondition(
                                        key="doc_id",
                                        match=models.MatchValue(
                                            value=replace_doc_id
                                        ),
                                    ),
                                    models.FieldCondition(
                                        key="active_generation",
                                        match=models.MatchValue(
                                            value=activate_generation
                                        ),
                                    ),
                                    models.FieldCondition(
                                        key="status",
                                        match=models.MatchValue(
                                            value="superseded"
                                        ),
                                    ),
                                ],
                                must_not=[
                                    models.FieldCondition(
                                        key="generation",
                                        match=models.MatchValue(
                                            value=activate_generation
                                        ),
                                    )
                                ],
                            )
                        )
                        if activate_generation
                        else models.PointIdsList(points=stale_ids)
                    )
                    delete_result = await client.delete(
                        collection_name=collection,
                        points_selector=selector,
                    )
                    _require_completed(delete_result, "stale cleanup")
                except Exception as exc:  # noqa: BLE001 - active generation is safe
                    if not activate_generation:
                        raise
                    logger.warning(
                        "Qdrant stale cleanup deferred for document '%s': %s",
                        replace_doc_id,
                        exc,
                    )
    except LLMError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _map_qdrant_error(exc) from exc
    return len(structs)


async def hybrid_search(
    dense: list[float],
    sparse_indices: list[int],
    sparse_values: list[float],
    top_k: int,
    candidates: int,
    query_filter: "models.Filter | None" = None,
    collection_name: str | None = None,
) -> list[dict]:
    """Dense+sparse retrieval fused with RRF; returns scored payloads.

    ``query_filter`` (optional) scopes both prefetch branches to a metadata
    subset — e.g. only physics textbook chunks for a physics lab.
    """
    collection = _resolve_collection_name(collection_name)
    client = get_client()
    base_filter = _without_inactive(query_filter)
    visible_filter = base_filter
    stale_generations: set[str] = set()
    stale_legacy_docs: set[str] = set()
    stale_point_ids: dict[str, object] = {}
    rows: list[dict] = []
    try:
        while True:
            result = await client.query_points(
                collection_name=collection,
                prefetch=[
                    models.Prefetch(
                        query=dense,
                        using=DENSE,
                        limit=candidates,
                        filter=visible_filter,
                    ),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse_indices, values=sparse_values
                        ),
                        using=SPARSE,
                        limit=candidates,
                        filter=visible_filter,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=max(top_k, candidates),
                with_payload=True,
            )
            rows = []
            hidden = False
            before = (
                len(stale_generations),
                len(stale_legacy_docs),
                len(stale_point_ids),
            )
            for point in result.points:
                payload = dict(point.payload or {})
                if _is_visible_payload(payload):
                    rows.append({"score": point.score, "payload": payload})
                    continue
                hidden = True
                generation = _payload_generation(payload)
                doc_id = payload.get("doc_id")
                if generation:
                    stale_generations.add(generation)
                elif isinstance(doc_id, str) and doc_id:
                    stale_legacy_docs.add(doc_id)
                else:
                    stale_point_ids[_point_id_key(point.id)] = point.id

            if len(rows) >= top_k or not hidden:
                break
            after = (
                len(stale_generations),
                len(stale_legacy_docs),
                len(stale_point_ids),
            )
            if after == before:
                break
            visible_filter = _with_generation_exclusions(
                base_filter,
                stale_generations,
                stale_legacy_docs,
                list(stale_point_ids.values()),
            )
    except Exception as exc:  # noqa: BLE001
        raise _map_qdrant_error(exc) from exc
    return rows[:top_k]


def _payload_generation(payload: dict) -> str | None:
    generation = payload.get("generation")
    return generation if isinstance(generation, str) and generation else None


def _has_inactive_status(payload: dict) -> bool:
    status = payload.get("status")
    return isinstance(status, str) and status.lower() in {
        "pending",
        "staging",
        "superseded",
    }


def _is_visible_payload(payload: dict) -> bool:
    if _has_inactive_status(payload):
        return False
    generation = _payload_generation(payload)
    active = payload.get("active_generation")
    if generation is None:
        return not (isinstance(active, str) and active)
    return generation == active


def _ordered_lab_payloads(payloads: list[dict]) -> list[dict]:
    if all(
        isinstance(payload.get("char_start"), int) and isinstance(payload.get("char_end"), int)
        for payload in payloads
    ):
        return sorted(
            payloads,
            key=lambda payload: (
                payload["char_start"],
                payload["char_end"],
                payload.get("chunk_index", 0),
                payload.get("text", ""),
            ),
        )
    return sorted(
        payloads,
        key=lambda payload: (
            payload.get("chunk_index", 0),
            payload.get("text", ""),
        ),
    )


def _reconstruct_lab_text(payloads: list[dict]) -> str:
    ordered = _ordered_lab_payloads(payloads)
    if not ordered:
        return ""
    if not all(
        isinstance(payload.get("char_start"), int) and isinstance(payload.get("char_end"), int)
        for payload in ordered
    ):
        return "\n".join(payload.get("text", "") for payload in ordered if payload.get("text")).strip()

    if any(
        payload["char_start"] < 0
        or payload["char_end"] < payload["char_start"]
        or len(payload.get("text", ""))
        > payload["char_end"] - payload["char_start"]
        for payload in ordered
    ):
        return "\n".join(
            payload.get("text", "") for payload in ordered if payload.get("text")
        ).strip()

    document_end = max(payload["char_end"] for payload in ordered)
    canvas: list[str | None] = [None] * document_end
    for payload in ordered:
        text = payload.get("text", "")
        if not text:
            continue
        start = payload["char_start"]
        width = payload["char_end"] - start
        extra = width - len(text)
        shifts = [0] if start == 0 else range(extra + 1)
        candidates = []
        for shift in shifts:
            window = " " * shift + text + " " * (extra - shift)
            conflicts = sum(
                canvas[start + index] not in (None, char)
                for index, char in enumerate(window)
            )
            matches = sum(
                canvas[start + index] == char for index, char in enumerate(window)
            )
            candidates.append((conflicts, -matches, window))
        best_score = min(candidate[:2] for candidate in candidates)
        best = [candidate[2] for candidate in candidates if candidate[:2] == best_score]
        if len(best) != 1:
            return "\n".join(
                item.get("text", "") for item in ordered if item.get("text")
            ).strip()
        window = best[0]
        for index, char in enumerate(window):
            position = start + index
            if canvas[position] is None:
                canvas[position] = char
    return "".join(char or " " for char in canvas).strip()


async def fetch_lab_instruction_record(
    lab_id: str, collection_name: str | None = None
) -> dict | None:
    """Return procedure text plus its stored chunk payloads for ``lab_id``.

    Scrolls all chunks tagged with this ``lab_id`` and concatenates them by
    ``chunk_index``. The payloads let callers cite the actual lab instruction
    document instead of only unrelated theory retrieval. Returns ``None`` when
    the lab has no instruction in the store.
    """
    collection = _resolve_collection_name(collection_name)
    client = get_client()
    flt = _without_inactive(
        models.Filter(
            must=[
                models.FieldCondition(
                    key="lab_id", match=models.MatchValue(value=lab_id)
                )
            ]
        )
    )
    try:
        if not await client.collection_exists(collection):
            return None
        rows: list[dict] = []
        offset = None
        while True:
            records, offset = await client.scroll(
                collection_name=collection,
                scroll_filter=flt,
                limit=_SCROLL_PAGE,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for rec in records:
                payload = dict(rec.payload or {})
                if not _is_visible_payload(payload):
                    continue
                rows.append(payload)
            if offset is None:
                break
    except Exception as exc:  # noqa: BLE001
        raise _map_qdrant_error(exc) from exc

    docs: dict[str, list[dict]] = {}
    for payload in rows:
        doc_id = payload.get("doc_id")
        if not doc_id:
            continue
        docs.setdefault(doc_id, []).append(payload)

    if len(docs) != 1:
        return None
    ordered = _ordered_lab_payloads(next(iter(docs.values())))
    text = _reconstruct_lab_text(ordered)
    return {"text": text, "payloads": ordered} if text else None


async def fetch_lab_instruction(
    lab_id: str, collection_name: str | None = None
) -> str:
    """Return only the full procedure text for ``lab_id``.

    This preserves the original public behavior for callers that do not need
    source metadata. New citation-aware callers should use
    :func:`fetch_lab_instruction_record`.
    """
    collection = _resolve_collection_name(collection_name)
    record = await fetch_lab_instruction_record(lab_id, collection_name=collection)
    return record["text"] if record else ""


async def list_documents(collection_name: str | None = None) -> list[dict]:
    """Group all chunks by ``payload["doc_id"]`` into one entry per document.

    Returns ``[]`` if the collection does not exist yet.
    """
    collection = _resolve_collection_name(collection_name)
    client = get_client()
    try:
        if not await client.collection_exists(collection):
            return []
        docs: dict[str, dict] = {}
        metadata_fields = (
            "doc_type",
            "source_type",
            "source_path",
            "subject",
            "grade",
            "lang",
            "lab_id",
            "lab_number",
            "file_type",
        )
        offset = None
        while True:
            records, offset = await client.scroll(
                collection_name=collection,
                limit=_SCROLL_PAGE,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for rec in records:
                payload = dict(rec.payload or {})
                if not _is_visible_payload(payload):
                    continue
                doc_id = payload.get("doc_id")
                if doc_id is None:
                    continue
                entry = docs.get(doc_id)
                if entry is None:
                    entry = {
                        "file_id": doc_id,
                        "filename": payload.get("filename"),
                        "chunks": 1,
                        "status": "ready",
                    }
                    for field in metadata_fields:
                        entry[field] = payload.get(field)
                    # Older ingests used ``source`` / ``doc_type`` before the
                    # explicit citation aliases were added. Expose useful
                    # listing metadata for those documents too.
                    entry["source_path"] = entry["source_path"] or payload.get("source")
                    entry["source_type"] = entry["source_type"] or entry["doc_type"]
                    docs[doc_id] = entry
                else:
                    entry["chunks"] += 1
                    if entry["filename"] is None:
                        entry["filename"] = payload.get("filename")
                    for field in metadata_fields:
                        if entry[field] is None:
                            entry[field] = payload.get(field)
                    entry["source_path"] = entry["source_path"] or payload.get("source")
                    entry["source_type"] = entry["source_type"] or entry["doc_type"]
            if offset is None:
                break
    except Exception as exc:  # noqa: BLE001
        raise _map_qdrant_error(exc) from exc
    return list(docs.values())


async def delete_document(
    doc_id: str, collection_name: str | None = None
) -> bool:
    """Delete every chunk for ``doc_id``; return False if there were none."""
    collection = _resolve_collection_name(collection_name)
    client = get_client()
    flt = _doc_filter(doc_id)
    try:
        count = (
            await client.count(
                collection_name=collection,
                count_filter=flt,
                exact=True,
            )
        ).count
        if count == 0:
            return False
        result = await client.delete(
            collection_name=collection,
            points_selector=models.FilterSelector(filter=flt),
        )
        _require_completed(result, "delete")
    except LLMError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _map_qdrant_error(exc) from exc
    logger.info("Deleted document '%s' (%d chunks)", doc_id, count)
    return True


async def collection_status(collection_name: str | None = None) -> dict:
    """Summarise the collection: status, point count and distinct documents."""
    collection = _resolve_collection_name(collection_name)
    client = get_client()
    try:
        if not await client.collection_exists(collection):
            return {
                "status": "unconfigured",
                "collection": collection,
                "points": 0,
                "documents": 0,
                "file_counts": {"total": 0},
                "documents_by_language": {
                    language: 0 for language in SUPPORTED_LANGUAGES
                },
                "supported_languages": list(SUPPORTED_LANGUAGES),
            }
        points = (
            await client.count(collection_name=collection, exact=True)
        ).count
    except Exception as exc:  # noqa: BLE001
        raise _map_qdrant_error(exc) from exc

    document_rows = await list_documents(collection_name=collection)
    documents = len(document_rows)
    documents_by_language = {
        language: sum(1 for row in document_rows if row.get("lang") == language)
        for language in SUPPORTED_LANGUAGES
    }
    return {
        "status": "ready" if points else "empty",
        "collection": collection,
        "points": points,
        "documents": documents,
        "file_counts": {"total": documents},
        "documents_by_language": documents_by_language,
        "supported_languages": list(SUPPORTED_LANGUAGES),
    }
