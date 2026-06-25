"""Knowledge-base ingestion into the local Qdrant vector store (hybrid RAG).

PDF/DOCX/EPUB/TXT/MD files are normalised to Markdown, chunked, embedded and
upserted *locally*: ``to_markdown`` converts via markitdown (falling back to
``pypdf`` / ``python-docx`` / a tiny EPUB parser), the text is split into
overlapping windows, embedded by the bge-m3 sidecar (dense + sparse) and written
to Qdrant. There is no hosted vector store — the KB lives in the Qdrant
collection.

The full ingest path is::

    upload_document
      -> to_markdown    (normalise pdf/docx/epub/txt/md -> Markdown)
      -> _chunk_text    (sliding character windows, CHUNK_SIZE/CHUNK_OVERLAP)
      -> embeddings.embed_texts   (bge-m3 dense + sparse)
      -> vectorstore.upsert_points   (payload carries lab metadata)

``bulk_ingest_tree`` walks the school corpus folder, derives per-file metadata
from the path (``corpus_meta.parse_path`` -> subject/grade/lang/lab_id) and
ingests each file with that metadata so retrieval can be scoped by lab context.
The document id is a stable ``uuid5`` of the ``doc_key`` (the relative path for
bulk ingest, since lab filenames repeat across grades), and existing chunks for
that id are deleted before re-upserting.

Validation failures (unsupported extension, etc.) raise ``ValueError``; any
embedding/vector-store failure surfaces as an ``LLMError`` subclass and is left
to propagate so routes can map it to 504/502.
"""

import io
import json
import logging
import posixpath
import re
import uuid
import xml.etree.ElementTree as ET
import zipfile
from html import unescape as _html_unescape
from pathlib import Path
from urllib.parse import unquote

import docx
import pypdf

from app.core.config import settings
from app.services import corpus_meta, embeddings, vectorstore

logger = logging.getLogger("assistant.ingestion")

# File types we can parse and embed locally for the school knowledge base.
# Everything is normalised to Markdown before chunking (see ``to_markdown``).
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".epub"}

# uuid5 namespace for deriving stable document/chunk ids from doc keys.
_DOC_NAMESPACE = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")

# Documents shorter than this many characters of extracted text are flagged as
# "stub" in the manifest (present but too thin to be a real lab procedure).
_STUB_CHARS = 200

_TAG_RE = re.compile(r"<[^>]+>")
# Drop <script>/<style> bodies before stripping tags so their contents never
# leak into the extracted text.
_SCRIPT_STYLE_RE = re.compile(r"(?is)<(script|style)\b[^>]*>.*?</\1>")
# A "word" for the image-only heuristic: a run of 2+ Cyrillic letters (the whole
# block, so both Russian and Kazakh). The school corpus is entirely Cyrillic, so
# a multi-MB EPUB that yields ~no Cyrillic words is a scanned/image-only book —
# its only "text" is stray Latin ids / page numbers left over from the markup,
# which would otherwise be indexed as a handful of useless noise chunks.
_WORD_RE = re.compile(r"[Ѐ-ӿ]{2,}")

# EPUB extraction thresholds (word counts via ``_WORD_RE``):
#   * if markitdown returns at least this many words we trust it and skip the
#     (more expensive) spine reparse;
#   * below ``_EPUB_MIN_WORDS`` the best extraction is treated as image-only
#     (scanned) and the file is skipped rather than indexed as a few noise
#     chunks of stripped <img> tags.
_EPUB_TRUST_WORDS = 500
_EPUB_MIN_WORDS = 50


def _doc_id(doc_key: str) -> str:
    """Stable document id for ``doc_key`` (re-ingest replaces in place).

    ``doc_key`` is the bare filename for single uploads but the *relative path*
    for bulk corpus ingest, since lab filenames (e.g. 'Лабораторная работа
    №2.docx') repeat across subjects/grades and would otherwise collide.
    """
    return uuid.uuid5(_DOC_NAMESPACE, doc_key).hex


def _markitdown(suffix: str, content: bytes) -> str | None:
    """Convert ``content`` to Markdown via markitdown, or None if unavailable.

    markitdown (optional dep) handles pdf/docx/epub/xlsx richly; when it is not
    installed or fails we fall back to the per-format extractors below.
    """
    try:
        from markitdown import MarkItDown
    except ImportError:
        return None
    try:
        result = MarkItDown().convert_stream(io.BytesIO(content), file_extension=suffix)
        text = (result.text_content or "").strip()
        return text or None
    except Exception as exc:  # noqa: BLE001 - any converter failure -> fallback
        logger.warning("markitdown failed for '%s' (%s); using fallback", suffix, exc)
        return None


def _count_words(text: str) -> int:
    """Number of word-ish tokens in ``text`` (cheap image-only heuristic)."""
    return len(_WORD_RE.findall(text))


def _html_to_text(html: str) -> str:
    """Strip an (x)html document to readable text (drops script/style, entities)."""
    html = _SCRIPT_STYLE_RE.sub(" ", html)
    text = _TAG_RE.sub(" ", html)
    return _html_unescape(text)


def _zip_join(base: str, href: str) -> str:
    """Resolve an OPF-relative href to a normalised zip member path."""
    href = unquote(href).split("#", 1)[0]  # drop url-encoding + fragment
    joined = f"{base}/{href}" if base else href
    return posixpath.normpath(joined)


def _opf_path(z: zipfile.ZipFile, names: list[str]) -> str | None:
    """Locate the OPF package document via META-INF/container.xml (or first .opf)."""
    if "META-INF/container.xml" in names:
        try:
            root = ET.fromstring(z.read("META-INF/container.xml"))
            for el in root.iter():
                if el.tag.rsplit("}", 1)[-1] == "rootfile":
                    full_path = el.get("full-path")
                    if full_path:
                        return full_path
        except ET.ParseError:
            pass
    for name in names:
        if name.lower().endswith(".opf"):
            return name
    return None


def _spine_docs(z: zipfile.ZipFile, opf_path: str) -> list[str]:
    """Return spine document zip paths in reading order, resolved against the OPF."""
    try:
        opf = ET.fromstring(z.read(opf_path))
    except (KeyError, ET.ParseError):
        return []
    manifest: dict[str, str] = {}
    for el in opf.iter():
        if el.tag.rsplit("}", 1)[-1] == "item":
            item_id, href = el.get("id"), el.get("href")
            if item_id and href:
                manifest[item_id] = href
    base = opf_path.rsplit("/", 1)[0] if "/" in opf_path else ""
    docs: list[str] = []
    for el in opf.iter():
        if el.tag.rsplit("}", 1)[-1] == "itemref":
            href = manifest.get(el.get("idref") or "")
            if href:
                docs.append(_zip_join(base, href))
    return docs


def _epub_to_text(content: bytes) -> str:
    """Robust EPUB -> text: follow the OPF spine, in reading order.

    markitdown (and ebooklib) sometimes return almost nothing for these school
    textbooks because the spine isn't followed, so we parse the OCF container
    ourselves: META-INF/container.xml -> OPF manifest+spine -> each spine
    document, tags stripped, concatenated in order. If the container/spine can't
    be read we fall back to every (x)html member, sorted alphabetically (the old
    behaviour).
    """
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        names = z.namelist()
        member_set = set(names)
        opf_path = _opf_path(z, names)
        docs = _spine_docs(z, opf_path) if opf_path else []
        docs = [d for d in docs if d in member_set]
        if not docs:
            docs = sorted(
                n for n in names if n.lower().endswith((".xhtml", ".html", ".htm"))
            )
        out: list[str] = []
        for name in docs:
            try:
                raw = z.read(name)
            except KeyError:
                continue
            out.append(_html_to_text(raw.decode("utf-8", "ignore")))
    return "\n".join(out)


def _extract_epub(filename: str, content: bytes) -> str:
    """EPUB -> text, resilient to markitdown silently extracting nothing.

    markitdown is tried first; when it returns suspiciously little text we
    re-parse the spine ourselves (:func:`_epub_to_text`) and keep whichever
    yields more words. Genuinely image-only (scanned) EPUBs extract ~no words —
    we log a clear warning naming the file and return ``""`` so the document is
    skipped (status ``empty``) rather than indexed as a few noise chunks. Those
    files are listed in the manifest as ``stub`` for the OCR decision.
    """
    md = _markitdown(".epub", content) or ""
    md_words = _count_words(md)

    best, best_words = md, md_words
    if md_words < _EPUB_TRUST_WORDS:
        try:
            spine = _epub_to_text(content)
        except Exception as exc:  # noqa: BLE001 - any zip/xml failure -> keep markitdown
            logger.warning("EPUB spine parse failed for '%s' (%s)", filename, exc)
            spine = ""
        if _count_words(spine) > best_words:
            best, best_words = spine, _count_words(spine)

    if best_words < _EPUB_MIN_WORDS:
        logger.warning(
            "EPUB '%s' yielded ~no extractable text (%d words) — likely "
            "image-only/scanned; skipping (needs OCR)",
            filename,
            best_words,
        )
        return ""
    return best


def to_markdown(filename: str, content: bytes) -> str:
    """Normalise any supported document to Markdown / plain text.

    Tries markitdown first (best fidelity, incl. tables and EPUB); otherwise
    falls back to pypdf / python-docx / a tiny EPUB tag-stripper. ``.md`` /
    ``.txt`` pass through unchanged. Raises ``ValueError`` for unsupported
    extensions.
    """
    suffix = Path(filename).suffix.lower()

    if suffix in (".txt", ".md"):
        return content.decode("utf-8", errors="replace")

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Supported: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    # EPUB has its own markitdown-first-then-spine-fallback path (markitdown
    # silently under-reads the spine on these textbooks), so handle it before the
    # generic markitdown attempt below.
    if suffix == ".epub":
        return _extract_epub(filename, content)

    md = _markitdown(suffix, content)
    if md is not None:
        return md

    if suffix == ".pdf":
        reader = pypdf.PdfReader(io.BytesIO(content))
        return "\n".join(p.extract_text() or "" for p in reader.pages)

    if suffix == ".docx":
        document = docx.Document(io.BytesIO(content))
        return "\n".join(p.text for p in document.paragraphs if p.text and p.text.strip())

    raise ValueError(f"Unsupported file type '{suffix}'")  # pragma: no cover


# Backwards-compatible alias (older callers / tests imported ``_extract_text``).
_extract_text = to_markdown


def _chunk_text(text: str) -> list[str]:
    """Split ``text`` into overlapping character windows.

    Whitespace is collapsed first, then windows of ``CHUNK_SIZE`` characters are
    emitted stepping by ``CHUNK_SIZE - CHUNK_OVERLAP``. Empty / whitespace-only
    windows are dropped.
    """
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    size = settings.CHUNK_SIZE
    overlap = settings.CHUNK_OVERLAP
    # Guard against a degenerate (or misconfigured) overlap >= size which would
    # make the window never advance.
    if overlap >= size:
        overlap = max(0, size // 4)
    step = size - overlap

    chunks: list[str] = []
    for start in range(0, len(normalized), step):
        window = normalized[start : start + size].strip()
        if window:
            chunks.append(window)
        if start + size >= len(normalized):
            break
    return chunks


async def upload_document(
    filename: str,
    content: bytes,
    metadata: dict | None = None,
    doc_key: str | None = None,
    **_,
) -> dict:
    """Convert to Markdown, chunk, embed and upsert one document into Qdrant.

    ``metadata`` (subject/grade/lang/doc_type/lab_id …) is merged into every
    chunk payload so retrieval can be filtered by lab context. ``doc_key`` is
    the stable identity for replace-semantics — defaults to ``filename`` for
    single uploads; bulk ingest passes the relative path to avoid collisions
    between same-named lab files.

    Returns ``{file_id, filename, status, chunks}``. ``status`` is ``"ready"``
    on success or ``"empty"`` if the document yielded no usable text. Raises
    ``ValueError`` for unsupported extensions; embedding / vector-store failures
    propagate as ``LLMError`` subclasses.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Supported: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    await vectorstore.ensure_collection()

    key = doc_key or filename
    doc_id = _doc_id(key)
    text = to_markdown(filename, content)
    chunks = _chunk_text(text)
    if not chunks:
        # No usable text (e.g. an image-only/scanned EPUB we now skip). Drop any
        # chunks a previous ingest stored for this document so stale noise does
        # not linger in the index — re-ingest must leave it genuinely empty.
        removed = await vectorstore.delete_document(doc_id)
        logger.info(
            "Ingested '%s' -> doc_id=%s status=empty (no text%s)",
            key, doc_id, "; removed stale chunks" if removed else "",
        )
        return {"file_id": doc_id, "filename": filename, "status": "empty", "chunks": 0}

    embeddings_list = await embeddings.embed_texts(chunks)

    # Replace semantics: drop any existing chunks for this document first. This
    # runs only after a successful embed, so a transient embedder failure leaves
    # the previously-indexed chunks intact rather than wiping the document.
    await vectorstore.delete_document(doc_id)

    base_payload = {k: v for k, v in (metadata or {}).items() if v is not None}
    points = [
        {
            "id": uuid.uuid5(_DOC_NAMESPACE, f"{key}:{i}").hex,
            "dense": emb.dense,
            "sparse_indices": emb.sparse_indices,
            "sparse_values": emb.sparse_values,
            "payload": {
                **base_payload,
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": i,
                "text": chunk,
            },
        }
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings_list))
    ]

    n = await vectorstore.upsert_points(points)
    logger.info("Ingested '%s' -> doc_id=%s status=ready chunks=%d", key, doc_id, n)
    return {"file_id": doc_id, "filename": filename, "status": "ready", "chunks": n}


# ── Bulk corpus ingest + manifest ────────────────────────────────────────────


def iter_corpus_files(root: str) -> list[Path]:
    """All supported files under ``root`` (recursive), sorted for determinism."""
    base = Path(root)
    return sorted(
        p
        for p in base.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


async def bulk_ingest_tree(root: str) -> dict:
    """Walk ``root``, derive metadata from each path and ingest every file.

    Returns a summary ``{root, total, ready, empty, skipped, errors, documents}``.
    Files whose path is not under a recognised corpus tier are skipped (logged).
    Per-file failures are collected rather than aborting the whole run.
    """
    files = iter_corpus_files(root)
    summary = {
        "root": root,
        "total": len(files),
        "ready": 0,
        "empty": 0,
        "skipped": 0,
        "errors": [],
        "documents": [],
    }
    for path in files:
        meta = corpus_meta.parse_path(str(path), corpus_root=root)
        if meta is None:
            summary["skipped"] += 1
            logger.info("Skip (unrecognised path): %s", path)
            continue
        try:
            result = await upload_document(
                path.name, path.read_bytes(), metadata=meta, doc_key=meta["source"]
            )
        except Exception as exc:  # noqa: BLE001 - keep going, record the failure
            summary["errors"].append({"source": meta["source"], "error": str(exc)})
            logger.warning("Ingest failed for %s: %s", path, exc)
            continue
        summary["ready" if result["status"] == "ready" else "empty"] += 1
        summary["documents"].append({**meta, **result})
    logger.info(
        "Bulk ingest of %s: %d ready, %d empty, %d skipped, %d errors",
        root, summary["ready"], summary["empty"], summary["skipped"], len(summary["errors"]),
    )
    return summary


def build_manifest(root: str) -> dict:
    """Scan the corpus tree and report lab completeness (no embedding).

    Produces ``{labs: {lab_id: {...}}, missing_metadata: [...], textbooks: N}``.
    A lab is ``complete`` when its procedure file extracts enough text, ``stub``
    when present-but-thin, and we record which languages exist per
    subject/grade/number so a missing translation is visible. This is an offline
    report; the request path treats "no instruction in Qdrant" as incomplete.
    """
    labs: dict[str, dict] = {}
    missing_metadata: list[str] = []
    textbooks = 0

    for path in iter_corpus_files(root):
        meta = corpus_meta.parse_path(str(path), corpus_root=root)
        if meta is None:
            missing_metadata.append(str(path))
            continue
        if meta["doc_type"] == "textbook":
            textbooks += 1
            continue
        lab_id = meta.get("lab_id")
        if not lab_id:
            missing_metadata.append(meta["source"])
            continue
        try:
            text = to_markdown(path.name, path.read_bytes())
        except Exception:  # noqa: BLE001
            text = ""
        chars = len(re.sub(r"\s+", " ", text).strip())
        labs[lab_id] = {
            "lab_id": lab_id,
            "subject": meta["subject"],
            "grade": meta["grade"],
            "lang": meta["lang"],
            "lab_number": meta["lab_number"],
            "source": meta["source"],
            "chars": chars,
            "status": "complete" if chars >= _STUB_CHARS else "stub",
        }

    return {
        "labs": dict(sorted(labs.items())),
        "missing_metadata": sorted(missing_metadata),
        "textbooks": textbooks,
    }


def write_manifest(root: str, out_path: str) -> dict:
    """Build the manifest for ``root`` and write it to ``out_path`` as JSON."""
    manifest = build_manifest(root)
    Path(out_path).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


async def list_documents(**_) -> list[dict]:
    """List ingested documents (one entry per ``doc_id``)."""
    return await vectorstore.list_documents()


async def delete_document(file_id: str, **_) -> bool:
    """Delete a document's chunks; return False if it did not exist."""
    return await vectorstore.delete_document(file_id)


async def corpus_status(**_) -> dict:
    """Return the Qdrant collection's status, point count and document count."""
    return await vectorstore.collection_status()
