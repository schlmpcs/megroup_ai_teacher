"""Unit tests for the corpus-metadata layer added for the school lab corpus.

Covers path -> metadata derivation (``corpus_meta``), Markdown normalisation +
bulk ingest + manifest (``ingestion``), the metadata search filter
(``vectorstore``) and lab-aware answer generation (``llm``). Everything below
the proxy stays mocked — no Qdrant, embedder or markitdown required.
"""

from types import SimpleNamespace

import pytest

from app.services import corpus_meta, ingestion, llm, vectorstore
from app.services.embeddings import Embedding


# ── corpus_meta: path -> metadata ─────────────────────────────────────────────

LAB_ROOT = "Лабораторные физхимбио/Материалы лабок/Лабораторные работы"
BOOK_ROOT = "Лабораторные физхимбио/Материалы лабок/Школьный материал 7-11 класс 3 предмета"


def test_parse_lab_instruction_ru():
    path = f"{LAB_ROOT}/Физика/Физика 10 класс/рус/Лабораторная работа №2.docx"
    meta = corpus_meta.parse_path(path, corpus_root="Лабораторные физхимбио")
    assert meta["doc_type"] == "lab_instruction"
    assert meta["subject"] == "physics"
    assert meta["grade"] == 10
    assert meta["lang"] == "ru"
    assert meta["lab_number"] == 2
    assert meta["lab_id"] == "physics-10-ru-02"
    assert meta["source"].startswith("Материалы лабок/")


def test_parse_lab_instruction_kk_spaced_number():
    path = f"{LAB_ROOT}/Физика/Физика 7 класс/каз/Зертханалық жұмыс № 5.docx"
    meta = corpus_meta.parse_path(path)
    assert meta["subject"] == "physics"
    assert meta["grade"] == 7
    assert meta["lang"] == "kk"
    assert meta["lab_id"] == "physics-7-kk-05"


def test_parse_lab_lang_folder_typo_russ():
    # The corpus has a "русс" folder (typo) for Физика 8 — must still be ru.
    path = f"{LAB_ROOT}/Физика/Физика 8 класс/русс/Лабораторная работа № 3.docx"
    meta = corpus_meta.parse_path(path)
    assert meta["lang"] == "ru"
    assert meta["lab_id"] == "physics-8-ru-03"


def test_parse_lab_number_alternate_form():
    path = f"{LAB_ROOT}/Физика/Физика 10 класс/рус/Лабораторная работа №1 (№3).docx"
    meta = corpus_meta.parse_path(path)
    # Primary (first) number wins.
    assert meta["lab_number"] == 1
    assert meta["lab_id"] == "physics-10-ru-01"


def test_parse_textbook_pdf_grade_from_filename():
    path = f"{BOOK_ROOT}/Химия/каз/Химия 8 каз.pdf"
    meta = corpus_meta.parse_path(path)
    assert meta["doc_type"] == "textbook"
    assert meta["subject"] == "chemistry"
    assert meta["grade"] == 8
    assert meta["lang"] == "kk"
    assert "lab_id" not in meta


def test_parse_textbook_epub_ru():
    path = f"{BOOK_ROOT}/Биология/рус/Биология 9 класс.epub"
    meta = corpus_meta.parse_path(path)
    assert meta["doc_type"] == "textbook"
    assert meta["subject"] == "biology"
    assert meta["grade"] == 9
    assert meta["lang"] == "ru"


def test_parse_unrecognised_path_returns_none():
    assert corpus_meta.parse_path("some/random/file.pdf") is None


def test_compose_lab_id_without_number_is_none():
    assert corpus_meta.compose_lab_id("physics", 10, "ru", None) is None


# ── ingestion: to_markdown + bulk + manifest ─────────────────────────────────


def test_to_markdown_txt_passthrough():
    assert ingestion.to_markdown("a.txt", "привет".encode()) == "привет"


def test_to_markdown_unsupported_raises():
    with pytest.raises(ValueError):
        ingestion.to_markdown("a.xyz", b"data")


class _RecordingVS:
    def __init__(self):
        self.upserted = []
        self.deleted = []

    async def ensure_collection(self):
        pass

    async def delete_document(self, doc_id):
        self.deleted.append(doc_id)
        return False

    async def upsert_points(self, points):
        self.upserted.extend(points)
        return len(points)


def _build_corpus(tmp_path):
    lab = (
        tmp_path
        / "Материалы лабок"
        / "Лабораторные работы"
        / "Химия"
        / "Химия 7 класс"
        / "рус"
    )
    lab.mkdir(parents=True)
    (lab / "Лабораторная работа №1.docx").write_text("")  # overwritten below
    # Use .md so to_markdown passes through without markitdown / docx libs.
    (lab / "Лабораторная работа №1.md").write_text(
        "Тема: Разделение смесей. " * 30, encoding="utf-8"
    )
    (lab / "Лабораторная работа №2.md").write_text("стоп", encoding="utf-8")  # stub

    book = (
        tmp_path
        / "Материалы лабок"
        / "Школьный материал 7-11 класс 3 предмета"
        / "Химия"
        / "рус"
    )
    book.mkdir(parents=True)
    (book / "Химия 7 класс.md").write_text("Теория. " * 50, encoding="utf-8")
    # Remove the stray empty .docx so it doesn't trip the docx parser.
    (lab / "Лабораторная работа №1.docx").unlink()
    return tmp_path


async def test_bulk_ingest_tags_metadata(tmp_path, monkeypatch):
    root = _build_corpus(tmp_path)
    vs = _RecordingVS()

    async def _embed_texts(texts):
        return [Embedding(dense=[float(i)], sparse_indices=[i], sparse_values=[1.0])
                for i, _ in enumerate(texts)]

    monkeypatch.setattr(ingestion.embeddings, "embed_texts", _embed_texts)
    for name in ("ensure_collection", "delete_document", "upsert_points"):
        monkeypatch.setattr(ingestion.vectorstore, name, getattr(vs, name))

    summary = await ingestion.bulk_ingest_tree(str(root))

    assert summary["total"] == 3
    assert summary["ready"] >= 2
    assert summary["errors"] == []

    lab_chunks = [p for p in vs.upserted if p["payload"]["doc_type"] == "lab_instruction"]
    assert lab_chunks, "expected lab_instruction chunks"
    payload = lab_chunks[0]["payload"]
    assert payload["subject"] == "chemistry"
    assert payload["grade"] == 7
    assert payload["lang"] == "ru"
    assert payload["lab_id"] == "chemistry-7-ru-01"
    # doc_key is the relative path, so same-named labs across grades never collide.
    assert payload["doc_id"] == ingestion._doc_id(payload["source"])


def test_build_manifest_flags_stub(tmp_path):
    root = _build_corpus(tmp_path)
    manifest = ingestion.build_manifest(str(root))
    assert manifest["textbooks"] == 1
    labs = manifest["labs"]
    assert labs["chemistry-7-ru-01"]["status"] == "complete"
    assert labs["chemistry-7-ru-02"]["status"] == "stub"


# ── vectorstore: metadata filter ─────────────────────────────────────────────


def test_meta_filter_drops_none_fields():
    flt = vectorstore.meta_filter(doc_type="textbook", subject="physics", grade=None)
    keys = {c.key for c in flt.must}
    assert keys == {"doc_type", "subject"}


def test_meta_filter_all_none_is_none():
    assert vectorstore.meta_filter(subject=None) is None


# ── llm: lab-aware grounding ─────────────────────────────────────────────────


async def test_generate_answer_injects_lab_instruction(monkeypatch):
    captured = {}

    async def _embed_query(text):
        return Embedding(dense=[0.0], sparse_indices=[], sparse_values=[])

    async def _hybrid_search(dense, sparse_indices, sparse_values, top_k, candidates,
                             query_filter=None):
        captured["filter"] = query_filter
        return []

    async def _fetch_lab_instruction(lab_id):
        captured["lab_id"] = lab_id
        return "Тема: Кипение. Ход работы: нагрей воду."

    def _meta_filter(**fields):
        captured["filter_fields"] = fields
        return "FILTER_SENTINEL"

    async def _create(**kwargs):
        captured["instructions"] = kwargs["instructions"]
        return SimpleNamespace(
            output_text="Вода кипит при 100°C.",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
        )

    monkeypatch.setattr(llm.embeddings, "embed_query", _embed_query)
    monkeypatch.setattr(llm.vectorstore, "hybrid_search", _hybrid_search)
    monkeypatch.setattr(llm.vectorstore, "fetch_lab_instruction", _fetch_lab_instruction)
    monkeypatch.setattr(llm.vectorstore, "meta_filter", _meta_filter)
    monkeypatch.setattr(llm.client.responses, "create", _create)

    lab = {"subject": "physics", "grade": 8, "lang": "ru", "lab_number": 2,
           "lab_id": "physics-8-ru-02"}
    result = await llm.generate_answer("Когда кипит вода?", lab=lab)

    assert result.answer == "Вода кипит при 100°C."
    assert captured["lab_id"] == "physics-8-ru-02"
    assert captured["filter"] == "FILTER_SENTINEL"
    assert captured["filter_fields"] == {"doc_type": "textbook", "subject": "physics"}
    assert "Ход работы: нагрей воду." in captured["instructions"]


async def test_generate_answer_incomplete_lab_warns(monkeypatch):
    captured = {}

    async def _embed_query(text):
        return Embedding(dense=[0.0], sparse_indices=[], sparse_values=[])

    async def _hybrid_search(dense, sparse_indices, sparse_values, top_k, candidates,
                             query_filter=None):
        return []

    async def _fetch_lab_instruction(lab_id):
        return ""  # missing instruction -> incomplete lab

    async def _create(**kwargs):
        captured["instructions"] = kwargs["instructions"]
        return SimpleNamespace(
            output_text="ответ",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
        )

    monkeypatch.setattr(llm.embeddings, "embed_query", _embed_query)
    monkeypatch.setattr(llm.vectorstore, "hybrid_search", _hybrid_search)
    monkeypatch.setattr(llm.vectorstore, "fetch_lab_instruction", _fetch_lab_instruction)
    monkeypatch.setattr(llm.client.responses, "create", _create)

    lab = {"subject": "biology", "grade": 9, "lang": "ru", "lab_number": 7,
           "lab_id": "biology-9-ru-07"}
    await llm.generate_answer("вопрос", lab=lab)
    assert "инструкция" in captured["instructions"].lower()
    assert "недоступна" in captured["instructions"].lower()
