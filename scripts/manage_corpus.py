"""CLI to manage the local Qdrant knowledge base (hybrid dense+sparse RAG).

Usage:
  python -m scripts.manage_corpus create-collection
  python -m scripts.manage_corpus upload path/to/file.pdf [more.docx ...]
  python -m scripts.manage_corpus bulk-ingest [CORPUS_ROOT]   # walk tree, tag, embed
  python -m scripts.manage_corpus gen-manifest [CORPUS_ROOT] [--out labs.json]
  python -m scripts.manage_corpus list
  python -m scripts.manage_corpus status
  python -m scripts.manage_corpus delete <doc_id>

Requires the local Qdrant and bge-m3 embedder sidecar to be reachable via
QDRANT_URL / EMBEDDING_BASE_URL in the environment / .env. OPENAI_API_KEY is
NOT needed for corpus management anymore — the knowledge base lives entirely in
the local Qdrant collection.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from app.core.config import settings
from app.services import ingestion, vectorstore


async def _create_collection() -> None:
    await vectorstore.ensure_collection()
    print(
        f"Collection ready: {settings.QDRANT_COLLECTION}  "
        f"(QDRANT_URL={settings.QDRANT_URL})"
    )


async def _upload(paths: list[str]) -> None:
    for p in paths:
        path = Path(p)
        if not path.is_file():
            print(f"  SKIP (not a file): {p}", file=sys.stderr)
            continue
        # ensure_collection() is called inside upload_document, so no need here.
        result = await ingestion.upload_document(path.name, path.read_bytes())
        print(
            f"  {result['status']:>10}  chunks={result['chunks']:<4}  "
            f"{result['file_id']}  {result['filename']}"
        )


async def _list() -> None:
    for f in await ingestion.list_documents():
        print(
            f"  {f['status']:>10}  {f['file_id']}  chunks={f.get('chunks', 0):<4}  "
            f"{f.get('filename') or ''}"
        )


async def _status() -> None:
    print(await ingestion.corpus_status())


async def _delete(doc_id: str) -> None:
    ok = await ingestion.delete_document(doc_id)
    print("deleted" if ok else "not found", doc_id)


async def _bulk_ingest(root: str, ocr: bool = False, only: str | None = None) -> None:
    summary = await ingestion.bulk_ingest_tree(root, ocr=ocr, only=only)
    print(
        f"Bulk ingest of {summary['root']}: "
        f"{summary['ready']} ready, {summary['empty']} empty, "
        f"{summary['skipped']} skipped, {summary.get('filtered', 0)} filtered, "
        f"{len(summary['errors'])} errors (of {summary['total']} files)"
    )
    for err in summary["errors"]:
        print(f"  ERROR  {err['source']}: {err['error']}", file=sys.stderr)


def _gen_manifest(root: str, out: str) -> None:
    manifest = ingestion.write_manifest(root, out)
    labs = manifest["labs"]
    stub = sum(1 for v in labs.values() if v["status"] == "stub")
    print(
        f"Manifest written to {out}: {len(labs)} labs "
        f"({len(labs) - stub} complete, {stub} stub), "
        f"{manifest['textbooks']} textbook files, "
        f"{len(manifest['missing_metadata'])} unrecognised paths"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage the local Qdrant knowledge base")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("create-collection")

    p_upload = sub.add_parser("upload")
    p_upload.add_argument("paths", nargs="+")

    p_bulk = sub.add_parser("bulk-ingest", help="Walk a corpus tree and ingest all files")
    p_bulk.add_argument("root", nargs="?", default=settings.CORPUS_ROOT)
    p_bulk.add_argument(
        "--ocr",
        action="store_true",
        help="OCR scanned/image-only PDFs & EPUBs (server-side, opt-in; needs Tesseract)",
    )
    p_bulk.add_argument(
        "--only",
        default=None,
        help="Only ingest files whose path contains this substring "
        "(keeps the root so doc_ids stay stable, e.g. --only 'Биология/рус')",
    )

    p_manifest = sub.add_parser("gen-manifest", help="Report lab completeness (no embedding)")
    p_manifest.add_argument("root", nargs="?", default=settings.CORPUS_ROOT)
    p_manifest.add_argument("--out", default=settings.LABS_MANIFEST)

    sub.add_parser("list")
    sub.add_parser("status")

    p_delete = sub.add_parser("delete")
    p_delete.add_argument("doc_id")

    args = parser.parse_args()

    if args.cmd == "create-collection":
        asyncio.run(_create_collection())
    elif args.cmd == "upload":
        asyncio.run(_upload(args.paths))
    elif args.cmd == "bulk-ingest":
        asyncio.run(_bulk_ingest(args.root, ocr=args.ocr, only=args.only))
    elif args.cmd == "gen-manifest":
        _gen_manifest(args.root, args.out)
    elif args.cmd == "list":
        asyncio.run(_list())
    elif args.cmd == "status":
        asyncio.run(_status())
    elif args.cmd == "delete":
        asyncio.run(_delete(args.doc_id))


if __name__ == "__main__":
    main()
