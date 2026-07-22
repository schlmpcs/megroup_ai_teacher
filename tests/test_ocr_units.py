"""Unit tests for the opt-in OCR fallback in ``app.services.ingestion``.

Scanned RU biology textbooks (grades 7/8/9) are EPUBs/PDFs whose pages are
images with no text layer, so normal extraction yields ~nothing. When OCR is
enabled at ingest time we render the pages and run Tesseract instead of skipping.

These tests are fully hermetic: Tesseract / pypdfium2 / Pillow are NOT installed
in CI, and the OCR helpers import them lazily, so we monkeypatch the OCR seams
(``_ocr_pdf`` / ``_ocr_epub_images`` / ``_ocr_image``) and never touch a binary.
The image-member resolver is tested directly with synthetic (text) zip members.
"""

import io
import zipfile

from app.services import ingestion

# Russian prose long enough to clear the image-only word threshold.
_RU = (
    "Клетка является основной структурной и функциональной единицей всех живых "
    "организмов. Клеточная мембрана регулирует поступление веществ внутрь клетки "
    "и выводит наружу продукты обмена. Ядро хранит наследственную информацию. "
) * 4

_CONTAINER_XML = (
    '<?xml version="1.0"?>'
    '<container version="1.0" '
    'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
    "<rootfiles>"
    '<rootfile full-path="{opf}" media-type="application/oebps-package+xml"/>'
    "</rootfiles></container>"
)


def _opf(items: list[tuple[str, str]]) -> str:
    manifest = "".join(
        f'<item id="{iid}" href="{href}" media-type="application/xhtml+xml"/>'
        for iid, href in items
    )
    spine = "".join(f'<itemref idref="{iid}"/>' for iid, _ in items)
    return (
        '<?xml version="1.0"?>'
        '<package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
        f"<manifest>{manifest}</manifest><spine>{spine}</spine></package>"
    )


def _xhtml(body: str) -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
        f"{body}</body></html>"
    )


def _make_epub(files: dict[str, str], *, spine: list[tuple[str, str]] | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("mimetype", "application/epub+zip")
        opf_path = "OEBPS/content.opf"
        z.writestr("META-INF/container.xml", _CONTAINER_XML.format(opf=opf_path))
        if spine is not None:
            z.writestr(opf_path, _opf(spine))
        for name, content in files.items():
            z.writestr(name, content)
    return buf.getvalue()


# ── Tesseract language mapping ────────────────────────────────────────────────


def test_tesseract_lang_mapping():
    assert ingestion._tesseract_lang("ru") == "rus"
    assert ingestion._tesseract_lang("kk") == "kaz"
    assert ingestion._tesseract_lang("en") == "eng"
    assert ingestion._tesseract_lang(None) == "rus+kaz+eng"
    assert ingestion._tesseract_lang("") == "rus+kaz+eng"


# ── EPUB image-member resolution (no image decoding involved) ─────────────────


def test_epub_image_members_follow_spine_order_and_dedup():
    # Spine lists ch2 before ch1; image members must come out in that order, and
    # a repeated <img> reference is de-duplicated.
    files = {
        "OEBPS/ch1.xhtml": _xhtml('<img src="img/p1.png"/><img src="img/p2.png"/>'),
        "OEBPS/ch2.xhtml": _xhtml('<img src="img/p0.png"/><img src="img/p1.png"/>'),
        "OEBPS/img/p0.png": "x",
        "OEBPS/img/p1.png": "x",
        "OEBPS/img/p2.png": "x",
    }
    epub = _make_epub(files, spine=[("c2", "ch2.xhtml"), ("c1", "ch1.xhtml")])
    with zipfile.ZipFile(io.BytesIO(epub)) as z:
        members = ingestion._epub_image_members(z, z.namelist())
    assert members == ["OEBPS/img/p0.png", "OEBPS/img/p1.png", "OEBPS/img/p2.png"]


def test_epub_image_members_fallback_to_all_images_sorted():
    # No <img> tags resolve -> fall back to every image member, sorted by name.
    files = {
        "OEBPS/ch1.xhtml": _xhtml("<p>текст без картинок</p>"),
        "OEBPS/b.jpg": "x",
        "OEBPS/a.jpg": "x",
        "OEBPS/notes.txt": "ignore me",
    }
    epub = _make_epub(files, spine=[("c1", "ch1.xhtml")])
    with zipfile.ZipFile(io.BytesIO(epub)) as z:
        members = ingestion._epub_image_members(z, z.namelist())
    assert members == ["OEBPS/a.jpg", "OEBPS/b.jpg"]


# ── to_markdown OCR routing (OCR seams mocked) ────────────────────────────────


def _image_only_epub() -> bytes:
    body = "".join(f'<img src="page{i}.png" alt=""/>' for i in range(8))
    return _make_epub({"OEBPS/p.xhtml": _xhtml(body)}, spine=[("p", "p.xhtml")])


def test_epub_ocr_off_still_skipped(monkeypatch, caplog):
    # Default behaviour preserved: image-only EPUB returns "" when OCR is off,
    # and the OCR helper is never called.
    monkeypatch.setattr(ingestion, "_markitdown", lambda suffix, content: "Обложка")

    def _boom(*a, **k):
        raise AssertionError("OCR must not run when ocr=False")

    monkeypatch.setattr(ingestion, "_ocr_epub_images", _boom)
    with caplog.at_level("WARNING"):
        text = ingestion.to_markdown("Биология 7 класс.epub", _image_only_epub())
    assert text == ""
    assert any("image-only" in r.message for r in caplog.records)


def test_epub_ocr_on_recovers_text(monkeypatch):
    monkeypatch.setattr(ingestion, "_markitdown", lambda suffix, content: "Обложка")
    captured = {}

    def fake_ocr(content, lang):
        captured["lang"] = lang
        return _RU

    monkeypatch.setattr(ingestion, "_ocr_epub_images", fake_ocr)
    text = ingestion.to_markdown(
        "Биология 7 класс.epub", _image_only_epub(), ocr=True, lang="ru"
    )
    assert "Клеточная мембрана" in text
    assert ingestion._count_words(text) > 50
    assert captured["lang"] == "rus"  # ru -> rus


def test_epub_ocr_on_but_empty_result_still_skips(monkeypatch):
    # If OCR itself recovers ~nothing, the document is still skipped (empty).
    monkeypatch.setattr(ingestion, "_markitdown", lambda suffix, content: "Обложка")
    monkeypatch.setattr(ingestion, "_ocr_epub_images", lambda content, lang: "  \n ")
    text = ingestion.to_markdown(
        "Биология 7 класс.epub", _image_only_epub(), ocr=True, lang="ru"
    )
    assert text == ""


def test_pdf_ocr_on_recovers_text(monkeypatch):
    # markitdown returns a thin (sub-threshold) string for a scanned PDF; OCR on
    # renders + recovers the real text. We never build a real PDF: _ocr_pdf is
    # mocked, and the thin markitdown result keeps us off the pypdf path.
    monkeypatch.setattr(ingestion, "_markitdown", lambda suffix, content: "12 ")
    captured = {}

    def fake_ocr_pdf(content, lang):
        captured["lang"] = lang
        return _RU

    monkeypatch.setattr(ingestion, "_ocr_pdf", fake_ocr_pdf)
    text = ingestion.to_markdown(
        "Биология 8 каз.pdf", b"%PDF-fake", ocr=True, lang="kk"
    )
    assert "Клеточная мембрана" in text
    assert captured["lang"] == "kaz"  # kk -> kaz


def test_pdf_ocr_off_returns_thin_text(monkeypatch):
    # OCR off: the (thin) extracted text is returned as-is, no OCR call.
    monkeypatch.setattr(ingestion, "_markitdown", lambda suffix, content: "12 abc")

    def _boom(*a, **k):
        raise AssertionError("OCR must not run when ocr=False")

    monkeypatch.setattr(ingestion, "_ocr_pdf", _boom)
    text = ingestion.to_markdown("scan.pdf", b"%PDF-fake")
    assert text == "12 abc"


def test_pdf_ocr_kept_only_if_better(monkeypatch):
    # Real text already present (>= threshold) -> OCR is never attempted.
    rich = _RU
    monkeypatch.setattr(ingestion, "_markitdown", lambda suffix, content: rich)

    def _boom(*a, **k):
        raise AssertionError("OCR must not run when text is already present")

    monkeypatch.setattr(ingestion, "_ocr_pdf", _boom)
    text = ingestion.to_markdown("good.pdf", b"%PDF-fake", ocr=True, lang="ru")
    assert text == rich.strip()


def test_pdf_high_word_count_okulyk_text_triggers_ocr(monkeypatch):
    notice = (
        "Книга предоставлена исключительно в образовательных целях согласно "
        "Приказа Министра образования и науки Республики Казахстан от 17 мая "
        "2019 года № 217 Все учебники Казахстана ищите на сайтах OKULYK.COM и "
        "OKULYK.KZ"
    )
    corrupt = "\n".join(f"page{i}\n{notice}" for i in range(1, 31))
    assert ingestion._count_words(corrupt) > 50
    monkeypatch.setattr(ingestion, "_markitdown", lambda suffix, content: corrupt)
    called = []

    def fake_ocr(content, lang):
        called.append(lang)
        return f"| page1 | | 1 |\n{_RU}\nВсе учебники Казахстана на OKULYK.KZ"

    monkeypatch.setattr(ingestion, "_ocr_pdf", fake_ocr)
    text = ingestion.to_markdown("Химия 8.pdf", b"%PDF-fake", ocr=True, lang="kk")

    assert called == ["kaz"]
    assert "Клеточная мембрана" in text
    assert "OKULYK" not in text
    assert "page1" not in text


def test_pdf_unknown_extreme_repetition_triggers_ocr(monkeypatch):
    # This has many Cyrillic words and no known OKULYK marker, but almost no
    # information diversity. The old word-count-only check trusted it.
    corrupt = "служебная копия школьного архива. " * 100
    monkeypatch.setattr(ingestion, "_markitdown", lambda suffix, content: corrupt)
    monkeypatch.setattr(ingestion, "_ocr_pdf", lambda content, lang: _RU)

    text = ingestion.to_markdown("Химия 9.pdf", b"%PDF-fake", ocr=True, lang="ru")

    assert text == _RU.strip()


def test_pdf_bad_ocr_does_not_replace_cleaned_text_layer(monkeypatch):
    useful = (
        "Химиялық реакция кезінде бастапқы заттардан жаңа заттар түзіледі және "
        "атомдардың жалпы саны сақталады. "
    ) * 6
    notice = (
        "Все учебники Казахстана на OKULYK.KZ Книга предоставлена исключительно "
        "в образовательных целях"
    )
    corrupt = "\n".join([notice] * 30 + [useful])
    monkeypatch.setattr(ingestion, "_markitdown", lambda suffix, content: corrupt)
    monkeypatch.setattr(
        ingestion,
        "_ocr_pdf",
        lambda content, lang: "неразборчивая копия страницы. " * 100,
    )

    text = ingestion.to_markdown("Химия 10.pdf", b"%PDF-fake", ocr=True, lang="ru")

    assert "Химиялық реакция" in text
    assert "OKULYK" not in text
    assert "неразборчивая" not in text
