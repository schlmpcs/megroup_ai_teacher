"""Unit tests for EPUB text extraction in ``app.services.ingestion``.

The school corpus ships multi-MB Russian biology EPUBs whose text markitdown
silently under-reads (it returns ~0 words because it doesn't follow the spine).
These tests build tiny synthetic EPUBs in-memory (no binaries committed) and
assert the spine fallback recovers the text, that image-only EPUBs are skipped
rather than indexed as noise, and that good markitdown output is still trusted.
"""

import io
import zipfile

import pytest

from app.services import ingestion

# A chunk of Russian prose long enough to clear the image-only word threshold.
_RU = (
    "Клетка является основной структурной и функциональной единицей всех живых "
    "организмов. Клеточная мембрана регулирует поступление веществ внутрь клетки "
    "и выводит наружу продукты обмена. Ядро хранит наследственную информацию и "
    "управляет процессами синтеза белка в цитоплазме клетки. "
) * 2

_CONTAINER_XML = (
    '<?xml version="1.0"?>'
    '<container version="1.0" '
    'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
    "<rootfiles>"
    '<rootfile full-path="{opf}" media-type="application/oebps-package+xml"/>'
    "</rootfiles></container>"
)


def _opf(spine_items: list[tuple[str, str]]) -> str:
    """Build a minimal OPF: ``spine_items`` is a list of (item_id, href)."""
    manifest = "".join(
        f'<item id="{iid}" href="{href}" media-type="application/xhtml+xml"/>'
        for iid, href in spine_items
    )
    spine = "".join(f'<itemref idref="{iid}"/>' for iid, _ in spine_items)
    return (
        '<?xml version="1.0"?>'
        '<package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
        f"<manifest>{manifest}</manifest>"
        f'<spine>{spine}</spine>'
        "</package>"
    )


def _xhtml(body: str) -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<html xmlns="http://www.w3.org/1999/xhtml"><head>'
        "<style>.x{color:red}</style></head>"
        f"<body>{body}</body></html>"
    )


def _make_epub(
    files: dict[str, str],
    *,
    opf_dir: str = "OEBPS",
    spine: list[tuple[str, str]] | None = None,
    with_container: bool = True,
) -> bytes:
    """Zip a synthetic EPUB. ``files`` maps zip member path -> contents."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("mimetype", "application/epub+zip")
        opf_path = f"{opf_dir}/content.opf" if opf_dir else "content.opf"
        if with_container:
            z.writestr("META-INF/container.xml", _CONTAINER_XML.format(opf=opf_path))
        if spine is not None:
            z.writestr(opf_path, _opf(spine))
        for name, content in files.items():
            z.writestr(name, content)
    return buf.getvalue()


@pytest.fixture(autouse=True)
def _markitdown_thin(monkeypatch):
    """Default: simulate the production bug (markitdown returns ~nothing)."""
    monkeypatch.setattr(ingestion, "_markitdown", lambda suffix, content: "Обложка")


def test_spine_fallback_recovers_text_when_markitdown_thin():
    epub = _make_epub(
        {
            "OEBPS/ch1.xhtml": _xhtml(f"<p>{_RU}</p>"),
            "OEBPS/ch2.xhtml": _xhtml("<h1>Глава вторая фотосинтез растений листья</h1>"),
        },
        spine=[("c1", "ch1.xhtml"), ("c2", "ch2.xhtml")],
    )
    text = ingestion.to_markdown("Биология 11 класс.epub", epub)

    assert "Клеточная мембрана" in text
    assert "фотосинтез" in text
    assert ingestion._count_words(text) > 50
    # markitdown's thin "Обложка" must not be what we returned.
    assert text.strip() != "Обложка"


def test_spine_reading_order_is_respected():
    # hrefs are alphabetical b/a but the spine lists a before b — spine wins.
    epub = _make_epub(
        {
            "OEBPS/a.xhtml": _xhtml(f"<p>ПЕРВЫЙ {_RU}</p>"),
            "OEBPS/b.xhtml": _xhtml(f"<p>ВТОРОЙ {_RU}</p>"),
        },
        spine=[("b", "b.xhtml"), ("a", "a.xhtml")],
    )
    text = ingestion.to_markdown("book.epub", epub)
    assert text.index("ВТОРОЙ") < text.index("ПЕРВЫЙ")


def test_image_only_epub_skipped_and_logged(caplog):
    # xhtml carries only <img> references — stripping tags leaves ~no words.
    body = "".join(f'<img src="page{i}.png" alt=""/>' for i in range(7))
    epub = _make_epub(
        {"OEBPS/p.xhtml": _xhtml(body)},
        spine=[("p", "p.xhtml")],
    )
    with caplog.at_level("WARNING"):
        text = ingestion.to_markdown("Биология 7 класс.epub", epub)

    assert text == ""
    assert any("image-only" in r.message for r in caplog.records)
    assert any("Биология 7 класс.epub" in r.message for r in caplog.records)


def test_alphabetical_fallback_without_container_or_opf():
    # No container.xml / OPF -> fall back to every xhtml, alphabetical.
    epub = _make_epub(
        {
            "text/00.xhtml": _xhtml(f"<p>{_RU}</p>"),
            "text/01.xhtml": _xhtml("<p>дополнительный материал клетка ядро</p>"),
        },
        spine=None,
        with_container=False,
    )
    text = ingestion.to_markdown("book.epub", epub)
    assert "Клеточная мембрана" in text
    assert "дополнительный материал" in text


def test_good_markitdown_is_trusted(monkeypatch):
    # When markitdown returns plenty of text, skip the spine reparse entirely.
    rich = "слово " * 600
    monkeypatch.setattr(ingestion, "_markitdown", lambda suffix, content: rich)
    # Image-only epub body: if the spine were used this would be skipped.
    epub = _make_epub(
        {"OEBPS/p.xhtml": _xhtml('<img src="x.png"/>')},
        spine=[("p", "p.xhtml")],
    )
    text = ingestion.to_markdown("book.epub", epub)
    assert text == rich


def test_html_to_text_drops_script_and_style():
    html = "<style>.a{x}</style><p>Привет</p><script>alert(1)</script>"
    out = ingestion._html_to_text(html)
    assert "Привет" in out
    assert "alert" not in out
    assert "{x}" not in out


def test_pdf_cleanup_removes_okulyk_notice_and_page_artifacts(monkeypatch):
    book_text = (
        "Химиялық элементтердің периодтық жүйесі атом құрылысы мен элементтердің "
        "қасиеттері арасындағы байланысты көрсетеді."
    )
    notice = (
        "*Книга предоставлена исключительно в образовательных целях согласно "
        "Приказа Министра образования и науки Республики Казахстан от 17 мая "
        "2019 года № 217 Все учебники Казахстана ищите на сайтах OKULYK.COM и "
        "OKULYK.KZ*"
    )
    wrapped_notice = notice.replace(
        "согласно Приказа", "согласно\nПриказа"
    ).replace("Республики Казахстан", "Республики Казахстан\n")
    extracted = "\n".join(
        [wrapped_notice, "page64", "| page65 | | 65 |", "1 2 3 4 5", book_text]
    )
    monkeypatch.setattr(ingestion, "_markitdown", lambda suffix, content: extracted)

    text = ingestion.to_markdown("Химия 8.pdf", b"%PDF-fake")

    assert text == book_text
    assert "OKULYK" not in text
    assert "page64" not in text
    assert "1 2 3 4 5" not in text


def test_pdf_cleanup_preserves_repeated_textbook_paragraphs():
    paragraph = (
        "Атом ядросы протондар мен нейтрондардан тұрады, ал электрондар ядроны "
        "айнала қозғалады. Бұл модель химиялық байланыстарды, валенттілікті және "
        "заттардың реакцияға түсу заңдылықтарын түсіндіруге көмектеседі."
    )
    extracted = "\n".join([paragraph] * 8)

    cleaned = ingestion._clean_pdf_extraction(extracted)

    assert cleaned.count(paragraph) == 8
    assert not ingestion._is_low_quality_pdf_extraction(extracted, cleaned)


def test_empty_document_removes_stale_chunks(monkeypatch):
    # A doc that now extracts to nothing (image-only EPUB) must delete any
    # chunks a previous ingest left behind, and must not call the embedder.
    import asyncio

    deleted: list[str] = []

    async def fake_ensure():
        return None

    async def fake_delete(doc_id):
        deleted.append(doc_id)
        return True

    async def fake_embed(_chunks):
        raise AssertionError("embed_texts must not run for an empty document")

    monkeypatch.setattr(ingestion, "to_markdown", lambda f, c, **k: "")
    monkeypatch.setattr(ingestion.vectorstore, "ensure_collection", fake_ensure)
    monkeypatch.setattr(ingestion.vectorstore, "delete_document", fake_delete)
    monkeypatch.setattr(ingestion.embeddings, "embed_texts", fake_embed)

    result = asyncio.run(
        ingestion.upload_document("Биология 7 класс.epub", b"x", doc_key="bio7")
    )
    assert result["status"] == "empty"
    assert result["chunks"] == 0
    assert deleted == [ingestion._doc_id("bio7")]
