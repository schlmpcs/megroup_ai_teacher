"""CLI to manage the local Qdrant knowledge base (hybrid dense+sparse RAG).

Usage:
  python -m scripts.manage_corpus create-collection
  python -m scripts.manage_corpus upload path/to/file.pdf [more.docx ...]
  python -m scripts.manage_corpus bulk-ingest [CORPUS_ROOT] [--prune]
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
from app.services.assistant_profiles import (
    DEFAULT_ASSISTANT_TYPE,
    get_assistant_profile,
)
from app.services import ingestion, vectorstore


async def _create_collection(collection_name: str) -> None:
    await vectorstore.ensure_collection(collection_name=collection_name)
    print(
        f"Collection ready: {collection_name}  "
        f"(QDRANT_URL={settings.QDRANT_URL})"
    )


async def _upload(paths: list[str], collection_name: str) -> None:
    for p in paths:
        path = Path(p)
        if not path.is_file():
            print(f"  SKIP (not a file): {p}", file=sys.stderr)
            continue
        # ensure_collection() is called inside upload_document, so no need here.
        result = await ingestion.upload_document(
            path.name,
            path.read_bytes(),
            collection_name=collection_name,
        )
        print(
            f"  {result['status']:>10}  chunks={result['chunks']:<4}  "
            f"{result['file_id']}  {result['filename']}"
        )


async def _list(collection_name: str) -> None:
    documents = await ingestion.list_documents(collection_name=collection_name)
    for f in documents:
        print(
            f"  {f['status']:>10}  {f['file_id']}  chunks={f.get('chunks', 0):<4}  "
            f"lang={f.get('lang') or '-':<2}  {f.get('filename') or ''}"
        )


async def _status(collection_name: str) -> None:
    status = await ingestion.corpus_status(collection_name=collection_name)
    print(status)


async def _delete(doc_id: str, collection_name: str) -> None:
    ok = await ingestion.delete_document(doc_id, collection_name=collection_name)
    print("deleted" if ok else "not found", doc_id)


async def _bulk_ingest(
    root: str,
    collection_name: str,
    ocr: bool | None = None,
    only: str | None = None,
    prune: bool = False,
) -> dict:
    summary = await ingestion.bulk_ingest_tree(
        root,
        ocr=ocr,
        only=only,
        prune=prune,
        collection_name=collection_name,
    )
    print(
        f"Bulk ingest of {summary['root']}: "
        f"{summary['ready']} ready, {summary['empty']} empty, "
        f"{summary['skipped']} skipped, {summary.get('filtered', 0)} filtered, "
        f"{len(summary['errors'])} errors (of {summary['total']} files)"
    )
    for err in summary["errors"]:
        print(f"  ERROR  {err['source']}: {err['error']}", file=sys.stderr)
    return summary


def _gen_manifest(root: str, out: str) -> None:
    manifest = ingestion.write_manifest(root, out)
    labs = manifest["labs"]
    stub = sum(1 for v in labs.values() if v["status"] == "stub")
    print(
        f"Manifest written to {out}: {len(labs)} labs "
        f"({len(labs) - stub} complete, {stub} stub), "
        f"{manifest['textbooks']} textbook files, "
        f"coverage={manifest.get('textbooks_by_language', {})}, "
        f"{len(manifest['missing_metadata'])} unrecognised paths"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage the local Qdrant knowledge base")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_assistant_type_argument(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument(
            "--assistant-type",
            default=DEFAULT_ASSISTANT_TYPE,
            help="Trusted assistant profile to use",
        )

    p_create = sub.add_parser("create-collection")
    add_assistant_type_argument(p_create)

    p_upload = sub.add_parser("upload")
    p_upload.add_argument("paths", nargs="+")
    add_assistant_type_argument(p_upload)

    p_bulk = sub.add_parser("bulk-ingest", help="Walk a corpus tree and ingest all files")
    p_bulk.add_argument("root", nargs="?", default=None)
    p_bulk.add_argument(
        "--ocr",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="OCR scanned/image-only PDFs & EPUBs (server-side, opt-in; needs Tesseract)",
    )
    p_bulk.add_argument(
        "--only",
        default=None,
        help="Only ingest files whose path contains this substring "
        "(keeps the root so doc_ids stay stable, e.g. --only 'Биология/рус')",
    )
    p_bulk.add_argument(
        "--prune",
        action="store_true",
        help="Delete stored corpus documents absent from this complete snapshot",
    )
    add_assistant_type_argument(p_bulk)

    p_manifest = sub.add_parser("gen-manifest", help="Report lab completeness (no embedding)")
    p_manifest.add_argument("root", nargs="?", default=None)
    p_manifest.add_argument("--out", default=settings.LABS_MANIFEST)
    add_assistant_type_argument(p_manifest)

    p_list = sub.add_parser("list")
    add_assistant_type_argument(p_list)

    p_status = sub.add_parser("status")
    add_assistant_type_argument(p_status)

    p_delete = sub.add_parser("delete")
    p_delete.add_argument("doc_id")
    add_assistant_type_argument(p_delete)

    args = parser.parse_args()
    try:
        profile = get_assistant_profile(args.assistant_type)
    except ValueError as exc:
        parser.error(str(exc))

    if args.cmd == "create-collection":
        asyncio.run(_create_collection(profile.qdrant_collection))
    elif args.cmd == "upload":
        asyncio.run(_upload(args.paths, profile.qdrant_collection))
    elif args.cmd == "bulk-ingest":
        if args.prune and args.only is not None:
            parser.error("--prune cannot be combined with --only")
        root = profile.corpus_root if args.root is None else args.root
        summary = asyncio.run(
            _bulk_ingest(
                root,
                profile.qdrant_collection,
                ocr=settings.OCR_ENABLED if args.ocr is None else args.ocr,
                only=args.only,
                prune=args.prune,
            )
        )
        if summary.get("errors"):
            raise SystemExit(1)
    elif args.cmd == "gen-manifest":
        root = profile.corpus_root if args.root is None else args.root
        _gen_manifest(root, args.out)
    elif args.cmd == "list":
        asyncio.run(_list(profile.qdrant_collection))
    elif args.cmd == "status":
        asyncio.run(_status(profile.qdrant_collection))
    elif args.cmd == "delete":
        asyncio.run(_delete(args.doc_id, profile.qdrant_collection))


if __name__ == "__main__":
    main()
