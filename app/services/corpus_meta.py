"""Derive knowledge-base metadata from the corpus folder layout.

The school corpus ships as a directory tree whose *paths* already encode every
piece of metadata we need — subject, grade, language and (for lab procedures)
the lab number. Rather than tag files by hand, we parse the path:

    …/Лабораторные работы/Физика/Физика 10 класс/рус/Лабораторная работа №2.docx
      -> doc_type=lab_instruction subject=physics grade=10 lang=ru lab_number=2
      -> lab_id = "physics-10-ru-02"

    …/Школьный материал…/Биология/рус/Биология 9 класс.epub
      -> doc_type=textbook subject=biology grade=9 lang=ru   (no lab_number)

The ``lab_id`` is the canonical key the simulator sends per request (composed
from structured ``subject/grade/lang/lab_number`` fields) so the assistant knows
which lab is running. All functions here are pure (no I/O) so they are trivially
unit-testable.
"""

import re
import unicodedata
from pathlib import PurePath
from typing import Optional

# Folder markers that separate the two corpus tiers.
LAB_ROOT_MARKER = "Лабораторные работы"
TEXTBOOK_ROOT_MARKER = "Школьный материал"

# Russian subject folder name -> canonical (ascii) subject slug.
_SUBJECTS = {
    "физика": "physics",
    "химия": "chemistry",
    "биология": "biology",
}

# Language folder / filename token -> canonical lang code. "русс" is a real
# (typo'd) folder name in the corpus; "сынып"/"қаз" guard Kazakh spellings.
_LANGS = {
    "рус": "ru",
    "русс": "ru",
    "ру": "ru",
    "каз": "kk",
    "қаз": "kk",
}

_GRADE_RE = re.compile(r"(\d{1,2})")
_LABNUM_RE = re.compile(r"№?\s*(\d{1,2})")

_UPLOAD_DOC_TYPES = {"textbook", "lab_instruction"}
_UPLOAD_SUBJECTS = {"physics", "chemistry", "biology"}
_UPLOAD_LANGS = {"ru", "kk"}


def _norm(token: str) -> str:
    return unicodedata.normalize("NFKC", token).strip().lower()


def _subject_of(token: str) -> Optional[str]:
    """Map a folder token (e.g. 'Физика' or 'Физика 10 класс') to a subject."""
    low = _norm(token)
    for ru, slug in _SUBJECTS.items():
        if low.startswith(ru):
            return slug
    return None


def _lang_of(token: str) -> Optional[str]:
    """Map a folder/filename token to a language code, if it names one."""
    low = _norm(token)
    if low in _LANGS:
        return _LANGS[low]
    # textbook filenames embed the lang as a word, e.g. "Химия 8 каз.pdf".
    for word in re.split(r"[\s_]+", low):
        if word in _LANGS:
            return _LANGS[word]
    return None


def _grade_of(*tokens: str) -> Optional[int]:
    """First integer in 7..11 found across ``tokens`` (grade appears first)."""
    for token in tokens:
        for m in _GRADE_RE.finditer(token):
            n = int(m.group(1))
            if 7 <= n <= 11:
                return n
    return None


def _lab_number_of(filename: str) -> Optional[int]:
    """Lab number from a procedure filename.

    Handles 'Лабораторная работа №2', 'Зертханалық жұмыс № 5',
    'Лабораторная работа 1' and 'Лабораторная работа №1 (№3)' (takes the
    primary/first number). Returns None if no number is present.
    """
    stem = PurePath(filename).stem
    m = _LABNUM_RE.search(stem)
    return int(m.group(1)) if m else None


def compose_lab_id(
    subject: str, grade: int, lang: str, lab_number: Optional[int]
) -> Optional[str]:
    """Canonical lab id, e.g. ``physics-10-ru-02``.

    Returns None when ``lab_number`` is missing — without it there is no single
    lab to anchor to (the request is then treated as theory-only for that
    subject/grade).
    """
    if lab_number is None:
        return None
    return f"{subject}-{grade}-{lang}-{lab_number:02d}"


def _upload_basename(filename: str) -> str:
    """Return a safe, Unicode-preserving basename for an admin upload."""
    normalized = unicodedata.normalize("NFKC", str(filename)).replace("\\", "/")
    basename = PurePath(normalized).name.strip()
    basename = re.sub(r"[\x00-\x1f\x7f]", "_", basename)
    if not basename or basename in {".", ".."}:
        raise ValueError("filename must contain a valid basename")
    return basename


def _upload_int(value: object, field: str, minimum: int, maximum: int) -> int:
    """Validate an integer upload field without silently accepting booleans."""
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer from {minimum} to {maximum}")
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field} must be an integer from {minimum} to {maximum}"
        ) from exc
    if str(value).strip() != str(parsed) or not minimum <= parsed <= maximum:
        raise ValueError(f"{field} must be an integer from {minimum} to {maximum}")
    return parsed


def build_upload_metadata(
    filename: str,
    doc_type: Optional[str] = None,
    subject: Optional[str] = None,
    grade: Optional[int] = None,
    lang: Optional[str] = None,
    lab_number: Optional[int] = None,
) -> tuple[Optional[dict], Optional[str]]:
    """Validate structured admin-upload fields and build stable metadata.

    With no structured fields this preserves the legacy upload behaviour by
    returning ``(None, None)``. Structured documents get a deterministic
    virtual source below ``admin_uploads``; that source is also their document
    key, so re-uploading the same filename in the same metadata scope replaces
    the document while identical filenames in other scopes remain distinct.
    """
    structured = (doc_type, subject, grade, lang, lab_number)
    if all(value is None for value in structured):
        return None, None
    if doc_type is None:
        raise ValueError("doc_type is required when structured metadata is provided")

    doc_type = str(doc_type).strip().lower()
    if doc_type not in _UPLOAD_DOC_TYPES:
        raise ValueError("doc_type must be 'textbook' or 'lab_instruction'")

    if subject is None or grade is None or lang is None:
        raise ValueError(f"{doc_type} requires subject, grade and lang")
    subject = str(subject).strip().lower()
    lang = str(lang).strip().lower()
    if subject not in _UPLOAD_SUBJECTS:
        raise ValueError("subject must be physics, chemistry or biology")
    if lang not in _UPLOAD_LANGS:
        raise ValueError("lang must be ru or kk")
    grade = _upload_int(grade, "grade", 7, 11)

    basename = _upload_basename(filename)
    scope = ["admin_uploads", doc_type, subject, str(grade), lang]
    metadata: dict = {
        "doc_type": doc_type,
        "subject": subject,
        "grade": grade,
        "lang": lang,
    }

    if doc_type == "textbook":
        if lab_number is not None:
            raise ValueError("textbook does not accept lab_number")
    else:
        if lab_number is None:
            raise ValueError("lab_instruction requires lab_number")
        lab_number = _upload_int(lab_number, "lab_number", 1, 99)
        lab_id = compose_lab_id(subject, grade, lang, lab_number)
        metadata.update(lab_number=lab_number, lab_id=lab_id)
        scope.append(f"{lab_number:02d}")

    source = "/".join([*scope, basename])
    metadata["source"] = source
    return metadata, source


def parse_path(path: str, corpus_root: Optional[str] = None) -> Optional[dict]:
    """Derive metadata for one corpus file from its path.

    Returns a dict with ``doc_type`` and the fields it could resolve
    (``subject``, ``grade``, ``lang``, ``lab_number``, ``lab_id``), or ``None``
    if the path is not under a recognised corpus tier. ``corpus_root`` is only
    used to make the stored ``source`` key relative.
    """
    p = PurePath(path)
    parts = list(p.parts)

    def _find(marker: str) -> Optional[int]:
        for i, part in enumerate(parts):
            if _norm(part).startswith(_norm(marker)):
                return i
        return None

    lab_idx = _find(LAB_ROOT_MARKER)
    book_idx = _find(TEXTBOOK_ROOT_MARKER)

    if corpus_root:
        try:
            source = str(p.relative_to(corpus_root))
        except ValueError:
            source = str(p)
    else:
        source = str(p)

    meta: dict = {"source": source, "filename": p.name}

    if lab_idx is not None:
        # …/<marker>/<Subject>/<Subject N класс>/<lang>/<file>
        tail = parts[lab_idx + 1 :]
        subject = _subject_of(tail[0]) if len(tail) >= 1 else None
        grade = _grade_of(tail[1]) if len(tail) >= 2 else None
        lang = _lang_of(tail[2]) if len(tail) >= 3 else None
        lab_number = _lab_number_of(p.name)
        meta.update(
            doc_type="lab_instruction",
            subject=subject,
            grade=grade,
            lang=lang,
            lab_number=lab_number,
        )
        if subject and grade and lang:
            meta["lab_id"] = compose_lab_id(subject, grade, lang, lab_number)
        return meta

    if book_idx is not None:
        # …/<marker>/<Subject>/<lang>/<file with grade in name>
        tail = parts[book_idx + 1 :]
        subject = _subject_of(tail[0]) if len(tail) >= 1 else None
        lang = _lang_of(tail[1]) if len(tail) >= 2 else _lang_of(p.name)
        grade = _grade_of(p.name)
        meta.update(
            doc_type="textbook",
            subject=subject,
            grade=grade,
            lang=lang,
        )
        return meta

    return None
