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
