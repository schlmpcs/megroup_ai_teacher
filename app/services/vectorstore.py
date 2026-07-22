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


def _point_id_key(point_id) -> str:
    try:
        return uuid.UUID(str(point_id)).hex
    except ValueError:
        return str(point_id)


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


async def ensure_collection() -> None:
    """Create the dense+sparse collection if it does not already exist."""
    client = get_client()
    try:
        if await client.collection_exists(settings.QDRANT_COLLECTION):
            return
        await client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config={
                DENSE: models.VectorParams(
                    size=settings.EMBEDDING_DIM,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={SPARSE: models.SparseVectorParams()},
        )
        logger.info("Created Qdrant collection '%s'", settings.QDRANT_COLLECTION)
    except Exception as exc:  # noqa: BLE001 - normalise to service-layer errors
        raise _map_qdrant_error(exc) from exc


async def upsert_points(points: list[dict]) -> int:
    """Upsert hybrid points and return how many were written.

    Each point dict carries ``id``, a ``dense`` vector, ``sparse_indices`` /
    ``sparse_values`` for the sparse vector, and a ``payload``. Empty input is
    a no-op (no network call) and returns 0.
    """
    if not points:
        return 0
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
            payload=p["payload"],
        )
        for p in points
    ]
    doc_ids = {point["payload"].get("doc_id") for point in points}
    replace_doc_id = doc_ids.pop() if len(doc_ids) == 1 else None
    new_ids = {_point_id_key(point["id"]) for point in points}
    client = get_client()
    try:
        async with _upsert_lock:
            old_ids: dict[str, object] = {}
            if replace_doc_id:
                offset = None
                while True:
                    records, offset = await client.scroll(
                        collection_name=settings.QDRANT_COLLECTION,
                        scroll_filter=_doc_filter(replace_doc_id),
                        limit=_SCROLL_PAGE,
                        offset=offset,
                        with_payload=False,
                        with_vectors=False,
                    )
                    old_ids.update(
                        {_point_id_key(record.id): record.id for record in records}
                    )
                    if offset is None:
                        break

            await client.upsert(
                collection_name=settings.QDRANT_COLLECTION,
                points=structs,
            )

            stale_ids = [
                point_id for key, point_id in old_ids.items() if key not in new_ids
            ]
            if stale_ids:
                await client.delete(
                    collection_name=settings.QDRANT_COLLECTION,
                    points_selector=models.PointIdsList(points=stale_ids),
                )
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
) -> list[dict]:
    """Dense+sparse retrieval fused with RRF; returns scored payloads.

    ``query_filter`` (optional) scopes both prefetch branches to a metadata
    subset — e.g. only physics textbook chunks for a physics lab.
    """
    client = get_client()
    try:
        result = await client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            prefetch=[
                models.Prefetch(
                    query=dense, using=DENSE, limit=candidates, filter=query_filter
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_indices, values=sparse_values
                    ),
                    using=SPARSE,
                    limit=candidates,
                    filter=query_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise _map_qdrant_error(exc) from exc
    return [{"score": point.score, "payload": point.payload} for point in result.points]


def _is_pending_lab_payload(payload: dict) -> bool:
    status = payload.get("status")
    return isinstance(status, str) and status.lower() in {"pending", "staging"}


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

    parts: list[str] = []
    previous_end: int | None = None
    for payload in ordered:
        text = payload.get("text", "")
        if not text:
            continue
        start = payload["char_start"]
        if previous_end is None:
            parts.append(text)
        else:
            overlap = max(previous_end - start, 0)
            parts.append(text[min(overlap, len(text)):])
        previous_end = max(previous_end or payload["char_end"], payload["char_end"])
    return "".join(parts).strip()


async def fetch_lab_instruction_record(lab_id: str) -> dict | None:
    """Return procedure text plus its stored chunk payloads for ``lab_id``.

    Scrolls all chunks tagged with this ``lab_id`` and concatenates them by
    ``chunk_index``. The payloads let callers cite the actual lab instruction
    document instead of only unrelated theory retrieval. Returns ``None`` when
    the lab has no instruction in the store.
    """
    client = get_client()
    flt = models.Filter(
        must=[models.FieldCondition(key="lab_id", match=models.MatchValue(value=lab_id))]
    )
    try:
        if not await client.collection_exists(settings.QDRANT_COLLECTION):
            return None
        rows: list[dict] = []
        offset = None
        while True:
            records, offset = await client.scroll(
                collection_name=settings.QDRANT_COLLECTION,
                scroll_filter=flt,
                limit=_SCROLL_PAGE,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for rec in records:
                payload = dict(rec.payload or {})
                if _is_pending_lab_payload(payload):
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

    matches: list[dict] = []
    for doc_payloads in docs.values():
        ordered = _ordered_lab_payloads(doc_payloads)
        text = _reconstruct_lab_text(ordered)
        if text:
            matches.append({"text": text, "payloads": ordered})

    if len(matches) != 1:
        return None
    return matches[0]


async def fetch_lab_instruction(lab_id: str) -> str:
    """Return only the full procedure text for ``lab_id``.

    This preserves the original public behavior for callers that do not need
    source metadata. New citation-aware callers should use
    :func:`fetch_lab_instruction_record`.
    """
    record = await fetch_lab_instruction_record(lab_id)
    return record["text"] if record else ""


async def list_documents() -> list[dict]:
    """Group all chunks by ``payload["doc_id"]`` into one entry per document.

    Returns ``[]`` if the collection does not exist yet.
    """
    client = get_client()
    try:
        if not await client.collection_exists(settings.QDRANT_COLLECTION):
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
                collection_name=settings.QDRANT_COLLECTION,
                limit=_SCROLL_PAGE,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for rec in records:
                payload = rec.payload or {}
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


async def delete_document(doc_id: str) -> bool:
    """Delete every chunk for ``doc_id``; return False if there were none."""
    client = get_client()
    flt = _doc_filter(doc_id)
    try:
        count = (
            await client.count(
                collection_name=settings.QDRANT_COLLECTION,
                count_filter=flt,
                exact=True,
            )
        ).count
        if count == 0:
            return False
        await client.delete(
            collection_name=settings.QDRANT_COLLECTION,
            points_selector=models.FilterSelector(filter=flt),
        )
    except Exception as exc:  # noqa: BLE001
        raise _map_qdrant_error(exc) from exc
    logger.info("Deleted document '%s' (%d chunks)", doc_id, count)
    return True


async def collection_status() -> dict:
    """Summarise the collection: status, point count and distinct documents."""
    client = get_client()
    try:
        if not await client.collection_exists(settings.QDRANT_COLLECTION):
            return {
                "status": "unconfigured",
                "collection": settings.QDRANT_COLLECTION,
                "points": 0,
                "documents": 0,
                "file_counts": {"total": 0},
                "documents_by_language": {
                    language: 0 for language in SUPPORTED_LANGUAGES
                },
                "supported_languages": list(SUPPORTED_LANGUAGES),
            }
        points = (
            await client.count(collection_name=settings.QDRANT_COLLECTION, exact=True)
        ).count
    except Exception as exc:  # noqa: BLE001
        raise _map_qdrant_error(exc) from exc

    document_rows = await list_documents()
    documents = len(document_rows)
    documents_by_language = {
        language: sum(1 for row in document_rows if row.get("lang") == language)
        for language in SUPPORTED_LANGUAGES
    }
    return {
        "status": "ready" if points else "empty",
        "collection": settings.QDRANT_COLLECTION,
        "points": points,
        "documents": documents,
        "file_counts": {"total": documents},
        "documents_by_language": documents_by_language,
        "supported_languages": list(SUPPORTED_LANGUAGES),
    }
