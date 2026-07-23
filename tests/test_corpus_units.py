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


def test_parse_english_lab_and_textbook_layouts():
    lab = corpus_meta.parse_path(
        "Corpus/Laboratory works/Physics/Physics Grade 10/en/"
        "Lab work No. 2.docx",
        corpus_root="Corpus",
    )
    book = corpus_meta.parse_path(
        "Corpus/School materials/Biology/english/Biology Grade 9.epub",
        corpus_root="Corpus",
    )

    assert lab == {
        "source": "Laboratory works/Physics/Physics Grade 10/en/"
        "Lab work No. 2.docx",
        "filename": "Lab work No. 2.docx",
        "doc_type": "lab_instruction",
        "subject": "physics",
        "grade": 10,
        "lang": "en",
        "lab_number": 2,
        "lab_id": "physics-10-en-02",
    }
    assert book["doc_type"] == "textbook"
    assert book["subject"] == "biology"
    assert book["grade"] == 9
    assert book["lang"] == "en"
    eng_alias = corpus_meta.parse_path(
        "Corpus/Textbooks/Chemistry/eng/Chemistry Grade 8.pdf"
    )
    assert eng_alias["lang"] == "en"


def test_parse_unrecognised_path_returns_none():
    assert corpus_meta.parse_path("some/random/file.pdf") is None


def test_compose_lab_id_without_number_is_none():
    assert corpus_meta.compose_lab_id("physics", 10, "ru", None) is None


def test_build_upload_metadata_legacy_upload_is_unchanged():
    assert corpus_meta.build_upload_metadata("notes.pdf") == (None, None)


def test_build_upload_metadata_textbook_has_stable_scoped_key():
    metadata, doc_key = corpus_meta.build_upload_metadata(
        "../unsafe/Physics 8.pdf",
        doc_type="textbook",
        subject="physics",
        grade=8,
        lang="ru",
    )

    assert doc_key == "admin_uploads/textbook/physics/8/ru/Physics 8.pdf"
    assert metadata == {
        "source": doc_key,
        "doc_type": "textbook",
        "subject": "physics",
        "grade": 8,
        "lang": "ru",
    }


def test_build_upload_metadata_lab_instruction_composes_lab_id():
    metadata, doc_key = corpus_meta.build_upload_metadata(
        r"C:\uploads\Lab 2.docx",
        doc_type="lab_instruction",
        subject="chemistry",
        grade=10,
        lang="kk",
        lab_number=2,
    )

    assert doc_key == "admin_uploads/lab_instruction/chemistry-10-kk-02"
    assert metadata["source"] == (
        "admin_uploads/lab_instruction/chemistry/10/kk/02/Lab 2.docx"
    )
    assert metadata["lab_number"] == 2
    assert metadata["lab_id"] == "chemistry-10-kk-02"


def test_build_upload_metadata_accepts_english():
    metadata, doc_key = corpus_meta.build_upload_metadata(
        "Lab work No. 2.docx",
        doc_type="lab_instruction",
        subject="physics",
        grade=10,
        lang="en",
        lab_number=2,
    )
    assert metadata["lab_id"] == "physics-10-en-02"
    assert doc_key == "admin_uploads/lab_instruction/physics-10-en-02"


def test_build_upload_metadata_scopes_same_filename_without_collisions():
    _, physics_key = corpus_meta.build_upload_metadata(
        "book.pdf", "textbook", "physics", 7, "ru"
    )
    _, biology_key = corpus_meta.build_upload_metadata(
        "book.pdf", "textbook", "biology", 7, "ru"
    )
    _, physics_key_again = corpus_meta.build_upload_metadata(
        "book.pdf", "textbook", "physics", 7, "ru"
    )

    assert physics_key != biology_key
    assert physics_key == physics_key_again


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"subject": "physics"}, "doc_type is required"),
        ({"doc_type": "notes"}, "doc_type must be"),
        ({"doc_type": "textbook"}, "requires subject, grade and lang"),
        (
            {"doc_type": "textbook", "subject": "math", "grade": 8, "lang": "ru"},
            "subject must be",
        ),
        (
            {"doc_type": "textbook", "subject": "physics", "grade": 6, "lang": "ru"},
            "grade must be",
        ),
        (
            {"doc_type": "textbook", "subject": "physics", "grade": 8, "lang": "de"},
            "lang must be",
        ),
        (
            {
                "doc_type": "textbook",
                "subject": "physics",
                "grade": 8,
                "lang": "ru",
                "lab_number": 1,
            },
            "does not accept lab_number",
        ),
        (
            {
                "doc_type": "lab_instruction",
                "subject": "physics",
                "grade": 8,
                "lang": "ru",
            },
            "requires lab_number",
        ),
        (
            {
                "doc_type": "lab_instruction",
                "subject": "physics",
                "grade": 8,
                "lang": "ru",
                "lab_number": 100,
            },
            "lab_number must be",
        ),
    ],
)
def test_build_upload_metadata_rejects_invalid_structured_fields(kwargs, message):
    with pytest.raises(ValueError, match=message):
        corpus_meta.build_upload_metadata("document.pdf", **kwargs)


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

    async def ensure_collection(self, collection_name=None):
        pass

    async def delete_document(self, doc_id, collection_name=None):
        self.deleted.append(doc_id)
        return False

    async def upsert_points(self, points, collection_name=None):
        self.upserted.extend(points)
        return len(points)

    async def list_documents(self, collection_name=None):
        return []


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
    for name in (
        "ensure_collection",
        "delete_document",
        "upsert_points",
        "list_documents",
    ):
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


def test_manifest_reports_english_coverage(tmp_path):
    lab_dir = (
        tmp_path
        / "Laboratory works"
        / "Physics"
        / "Physics Grade 10"
        / "en"
    )
    book_dir = tmp_path / "School materials" / "Biology" / "en"
    lab_dir.mkdir(parents=True)
    book_dir.mkdir(parents=True)
    (lab_dir / "Lab work No. 2.md").write_text(
        "Purpose and procedure for heating water safely. " * 12,
        encoding="utf-8",
    )
    (book_dir / "Biology Grade 9.md").write_text(
        "Cells have membranes that regulate transport. " * 12,
        encoding="utf-8",
    )

    manifest = ingestion.build_manifest(str(tmp_path))

    assert manifest["labs"]["physics-10-en-02"]["status"] == "complete"
    assert manifest["labs_by_language"] == {"en": 1}
    assert manifest["textbooks_by_language"] == {"en": 1}


# ── vectorstore: metadata filter ─────────────────────────────────────────────


def test_meta_filter_drops_none_fields():
    flt = vectorstore.meta_filter(doc_type="textbook", subject="physics", grade=None)
    keys = {c.key for c in flt.must}
    assert keys == {"doc_type", "subject"}


def test_meta_filter_all_none_is_none():
    assert vectorstore.meta_filter(subject=None) is None


def test_with_lang_appends_lang_to_base():
    base = vectorstore.meta_filter(doc_type="textbook", subject="physics")
    flt = vectorstore.with_lang(base, "ru")
    conds = {c.key: c.match.value for c in flt.must}
    assert conds == {"doc_type": "textbook", "subject": "physics", "lang": "ru"}
    # base is left untouched (no lang condition leaked into it)
    assert {c.key for c in base.must} == {"doc_type", "subject"}


def test_with_lang_none_base_builds_lang_only_filter():
    flt = vectorstore.with_lang(None, "kk")
    assert [(c.key, c.match.value) for c in flt.must] == [("lang", "kk")]


def test_with_lang_no_lang_returns_base_unchanged():
    base = vectorstore.meta_filter(subject="biology")
    assert vectorstore.with_lang(base, None) is base
    assert vectorstore.with_lang(None, None) is None


async def test_fetch_lab_instruction_record_keeps_source_payloads(monkeypatch):
    payload_1 = {
        "doc_id": "lab-doc",
        "filename": "Лабораторная работа №2.docx",
        "doc_type": "lab_instruction",
        "source": "Физика 8 класс/рус/Лабораторная работа №2.docx",
        "lab_id": "physics-8-ru-02",
        "lab_number": 2,
        "chunk_index": 1,
        "text": "Шаг второй.",
    }
    payload_0 = {**payload_1, "chunk_index": 0, "text": "Шаг первый."}

    class _Client:
        async def collection_exists(self, name):
            return True

        async def scroll(self, **kwargs):
            return [SimpleNamespace(payload=payload_1), SimpleNamespace(payload=payload_0)], None

    monkeypatch.setattr(vectorstore, "get_client", lambda: _Client())

    record = await vectorstore.fetch_lab_instruction_record("physics-8-ru-02")

    assert record["text"] == "Шаг первый.\nШаг второй."
    assert [payload["chunk_index"] for payload in record["payloads"]] == [0, 1]
    assert record["payloads"][0]["source"].endswith("Лабораторная работа №2.docx")
    # The old public helper remains text-only for backwards compatibility.
    assert await vectorstore.fetch_lab_instruction("physics-8-ru-02") == record["text"]


async def test_list_documents_exposes_document_metadata(monkeypatch):
    first = {
        "doc_id": "physics-book-8",
        "filename": "Physics 8.pdf",
        "doc_type": "textbook",
        "source": "admin_uploads/textbook/physics/8/ru/Physics 8.pdf",
        "subject": "physics",
        "grade": 8,
        "lang": "ru",
        "file_type": "pdf",
    }
    second = {**first, "source_type": "textbook", "source_path": first["source"]}

    class _Client:
        async def collection_exists(self, name):
            return True

        async def scroll(self, **kwargs):
            return [SimpleNamespace(payload=first), SimpleNamespace(payload=second)], None

    monkeypatch.setattr(vectorstore, "get_client", lambda: _Client())

    documents = await vectorstore.list_documents()

    assert documents == [
        {
            "file_id": "physics-book-8",
            "filename": "Physics 8.pdf",
            "chunks": 2,
            "status": "ready",
            "doc_type": "textbook",
            "source_type": "textbook",
            "source_path": first["source"],
            "subject": "physics",
            "grade": 8,
            "lang": "ru",
            "lab_id": None,
            "lab_number": None,
            "file_type": "pdf",
        }
    ]


async def test_collection_status_reports_per_language_documents(monkeypatch):
    class _Client:
        async def collection_exists(self, name):
            return True

        async def count(self, **kwargs):
            return SimpleNamespace(count=7)

    async def _documents(collection_name=None):
        return [{"lang": "ru"}, {"lang": "en"}, {"lang": "en"}, {"lang": None}]

    monkeypatch.setattr(vectorstore, "get_client", lambda: _Client())
    monkeypatch.setattr(vectorstore, "list_documents", _documents)

    status = await vectorstore.collection_status()

    assert status["supported_languages"] == ["ru", "kk", "en"]
    assert status["documents_by_language"] == {"ru": 1, "kk": 0, "en": 2}


# ── llm: lab-aware grounding ─────────────────────────────────────────────────


async def test_generate_answer_injects_lab_instruction(monkeypatch):
    captured = {}

    async def _embed_query(text):
        return Embedding(dense=[0.0], sparse_indices=[], sparse_values=[])

    async def _hybrid_search(
        dense,
        sparse_indices,
        sparse_values,
        top_k,
        candidates,
        query_filter=None,
        collection_name=None,
    ):
        captured["filter"] = query_filter
        return [
            {
                "score": 0.9,
                "payload": {
                    "doc_id": "physics-book-8",
                    "filename": "Физика 8 класс.pdf",
                    "doc_type": "textbook",
                    "source": "Физика/рус/Физика 8 класс.pdf",
                    "chunk_index": 4,
                    "page_start": 52,
                    "page_end": 52,
                    "text": "Кипение происходит во всем объеме жидкости.",
                },
            }
        ]

    async def _fetch_lab_instruction_record(lab_id, collection_name=None):
        captured["lab_id"] = lab_id
        return {
            "text": "Тема: Кипение. Ход работы: нагрей воду.",
            "payloads": [
                {
                    "doc_id": "lab-doc-2",
                    "filename": "Лабораторная работа №2.docx",
                    "doc_type": "lab_instruction",
                    "source": "Физика 8 класс/рус/Лабораторная работа №2.docx",
                    "chunk_index": 0,
                    "lab_id": lab_id,
                    "lab_number": 2,
                }
            ],
        }

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
    monkeypatch.setattr(
        llm.vectorstore,
        "fetch_lab_instruction_record",
        _fetch_lab_instruction_record,
    )
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
    # This is a theory question, so the retrieved textbook remains primary even
    # though the lab instruction was also injected and cited.
    assert result.primary_source["file_id"] == "physics-book-8"
    assert result.primary_source["source_type"] == "textbook"
    assert result.citations[1]["file_id"] == "lab-doc-2"
    assert result.citations[1]["source_type"] == "lab_instruction"
    assert result.citations[1]["lab_id"] == "physics-8-ru-02"


async def test_generate_answer_incomplete_lab_warns(monkeypatch):
    captured = {}

    async def _embed_query(text):
        return Embedding(dense=[0.0], sparse_indices=[], sparse_values=[])

    async def _hybrid_search(
        dense,
        sparse_indices,
        sparse_values,
        top_k,
        candidates,
        query_filter=None,
        collection_name=None,
    ):
        return []

    async def _fetch_lab_instruction_record(lab_id, collection_name=None):
        return None  # missing instruction -> incomplete lab

    async def _create(**kwargs):
        captured["instructions"] = kwargs["instructions"]
        return SimpleNamespace(
            output_text="ответ",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
        )

    monkeypatch.setattr(llm.embeddings, "embed_query", _embed_query)
    monkeypatch.setattr(llm.vectorstore, "hybrid_search", _hybrid_search)
    monkeypatch.setattr(
        llm.vectorstore,
        "fetch_lab_instruction_record",
        _fetch_lab_instruction_record,
    )
    monkeypatch.setattr(llm.client.responses, "create", _create)

    lab = {"subject": "biology", "grade": 9, "lang": "ru", "lab_number": 7,
           "lab_id": "biology-9-ru-07"}
    await llm.generate_answer("вопрос", lab=lab)
    assert "exact instruction is unavailable" in captured["instructions"].lower()
    assert "CURRENT SUBJECT AND LABORATORY SCOPE" in captured["instructions"]
    assert llm._GENERAL_KNOWLEDGE_MARKER not in captured["instructions"]


# ── llm: language-priority retrieval ─────────────────────────────────────────


async def test_prepare_grounding_infers_subject_and_language_without_lab(monkeypatch):
    captured = {}

    async def _retrieve(
        query,
        query_filter=None,
        lang=None,
        fallback_filter=None,
        collection_name=None,
    ):
        captured["query_filter"] = query_filter
        captured["lang"] = lang
        captured["fallback_filter"] = fallback_filter
        return []

    monkeypatch.setattr(llm, "_retrieve", _retrieve)

    grounding = await llm._prepare_answer_grounding(
        "қандай атақты химиктер бар?", None, None, None
    )

    conditions = {
        condition.key: condition.match.value
        for condition in captured["query_filter"].must
    }
    assert conditions == {"doc_type": "textbook", "subject": "chemistry"}
    assert captured["lang"] == "kk"
    assert captured["fallback_filter"] is None
    assert grounding.allow_general_knowledge is True


def _chunk(lang, text, score=0.9):
    return {"score": score, "payload": {"doc_id": text, "filename": f"{text}.pdf",
                                        "chunk_index": 0, "text": text, "lang": lang}}


async def test_retrieve_prefers_same_language_no_fallback(monkeypatch):
    """Same-language pass is full -> no fallback search runs."""
    monkeypatch.setattr(llm.settings, "RETRIEVAL_TOP_K", 2, raising=False)
    calls = []

    async def _embed_query(text):
        return Embedding(dense=[0.0], sparse_indices=[], sparse_values=[])

    async def _hybrid_search(dense, si, sv, top_k, candidates, query_filter=None):
        # The lang condition is present on the (only) call.
        lang = next((c.match.value for c in query_filter.must if c.key == "lang"), None)
        calls.append(lang)
        return [_chunk("ru", "ru-1"), _chunk("ru", "ru-2")]

    monkeypatch.setattr(llm.embeddings, "embed_query", _embed_query)
    monkeypatch.setattr(llm.vectorstore, "hybrid_search", _hybrid_search)

    base = vectorstore.meta_filter(doc_type="textbook", subject="physics")
    chunks = await llm._retrieve("q", query_filter=base, lang="ru")

    assert calls == ["ru"]  # only the same-language search ran
    assert [c["payload"]["text"] for c in chunks] == ["ru-1", "ru-2"]


async def test_retrieve_falls_back_when_same_language_thin(monkeypatch):
    """Thin same-language pass -> unconstrained fallback backfills other lang."""
    monkeypatch.setattr(llm.settings, "RETRIEVAL_TOP_K", 3, raising=False)
    seen_filters = []

    async def _embed_query(text):
        return Embedding(dense=[0.0], sparse_indices=[], sparse_values=[])

    async def _hybrid_search(dense, si, sv, top_k, candidates, query_filter=None):
        has_lang = any(c.key == "lang" for c in query_filter.must)
        seen_filters.append("lang" if has_lang else "base")
        if has_lang:
            return [_chunk("ru", "ru-1")]  # only one same-language hit (thin)
        # fallback returns a mix; same-language dupes are skipped, kk backfills
        return [_chunk("ru", "ru-1"), _chunk("kk", "kk-1"), _chunk("kk", "kk-2")]

    monkeypatch.setattr(llm.embeddings, "embed_query", _embed_query)
    monkeypatch.setattr(llm.vectorstore, "hybrid_search", _hybrid_search)

    base = vectorstore.meta_filter(doc_type="textbook", subject="physics")
    chunks = await llm._retrieve("q", query_filter=base, lang="ru")

    assert seen_filters == ["lang", "base"]  # both passes ran
    texts = [c["payload"]["text"] for c in chunks]
    # same-language first, then other-language backfill, no ru-1 duplicate
    assert texts == ["ru-1", "kk-1", "kk-2"]


async def test_retrieve_prefers_english_then_cross_language_without_duplicates(
    monkeypatch,
):
    monkeypatch.setattr(llm.settings, "RETRIEVAL_TOP_K", 3, raising=False)
    seen = []

    async def _embed_query(text):
        return Embedding(dense=[0.0], sparse_indices=[], sparse_values=[])

    async def _hybrid_search(dense, si, sv, top_k, candidates, query_filter=None):
        language = next(
            (condition.match.value for condition in query_filter.must if condition.key == "lang"),
            None,
        )
        seen.append(language)
        if language == "en":
            return [_chunk("en", "en-1")]
        return [_chunk("en", "en-1"), _chunk("ru", "ru-1"), _chunk("kk", "kk-1")]

    monkeypatch.setattr(llm.embeddings, "embed_query", _embed_query)
    monkeypatch.setattr(llm.vectorstore, "hybrid_search", _hybrid_search)

    base = vectorstore.meta_filter(doc_type="textbook", subject="physics")
    chunks = await llm._retrieve("boiling", query_filter=base, lang="en")

    assert seen == ["en", None]
    assert [chunk["payload"]["text"] for chunk in chunks] == [
        "en-1",
        "ru-1",
        "kk-1",
    ]


async def test_retrieve_no_lang_single_search(monkeypatch):
    """Without a known lang, behaviour is unchanged (one search, no filter narrowing)."""
    calls = []

    async def _embed_query(text):
        return Embedding(dense=[0.0], sparse_indices=[], sparse_values=[])

    async def _hybrid_search(dense, si, sv, top_k, candidates, query_filter=None):
        calls.append(query_filter)
        return [_chunk("ru", "ru-1")]

    monkeypatch.setattr(llm.embeddings, "embed_query", _embed_query)
    monkeypatch.setattr(llm.vectorstore, "hybrid_search", _hybrid_search)

    chunks = await llm._retrieve("q", query_filter=None, lang=None)
    assert calls == [None]  # single search, filter passed through untouched
    assert [c["payload"]["text"] for c in chunks] == ["ru-1"]
