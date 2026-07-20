"""Russian abbreviation, initial, and letter normalization for TTS."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import count
from typing import Match, Pattern

from .formula_speech import speak_formula, speak_reaction


RUSSIAN_ABBREVIATIONS = {
    "т.е.": "то есть",
    "т.к.": "так как",
    "и т.д.": "и так далее",
    "и т.п.": "и тому подобное",
    "г.": "год",
    "стр.": "страница",
    "рис.": "рисунок",
    "табл.": "таблица",
}

RUSSIAN_UNIT_ABBREVIATIONS = {
    "кг": "килограмм",
    "г": "грамм",
    "мг": "миллиграмм",
    "мл": "миллилитр",
    "см": "сантиметр",
    "км": "километр",
    "мин": "минута",
    "ч": "час",
    "°C": "градус Цельсия",
}

RUSSIAN_ACRONYMS = {
    "AI": "эй ай",
    "VR": "ви ар",
    "API": "эй пи ай",
    "pH": "пэ аш",
    "DNA": "ди эн эй",
}

RUSSIAN_LETTER_NAMES = {
    "А": "а",
    "Б": "бэ",
    "В": "вэ",
    "Г": "гэ",
    "Д": "дэ",
    "Е": "е",
    "Ё": "ё",
    "Ж": "жэ",
    "З": "зэ",
    "И": "и",
    "Й": "и краткое",
    "К": "ка",
    "Л": "эль",
    "М": "эм",
    "Н": "эн",
    "О": "о",
    "П": "пэ",
    "Р": "эр",
    "С": "эс",
    "Т": "тэ",
    "У": "у",
    "Ф": "эф",
    "Х": "ха",
    "Ц": "цэ",
    "Ч": "че",
    "Ш": "ша",
    "Щ": "ща",
    "Ъ": "твёрдый знак",
    "Ы": "ы",
    "Ь": "мягкий знак",
    "Э": "э",
    "Ю": "ю",
    "Я": "я",
}

LATIN_LETTER_NAMES_RU = {
    "A": "эй",
    "B": "би",
    "C": "си",
    "D": "ди",
    "E": "и",
    "F": "эф",
    "G": "джи",
    "H": "эйч",
    "I": "ай",
    "J": "джей",
    "K": "кей",
    "L": "эл",
    "M": "эм",
    "N": "эн",
    "O": "оу",
    "P": "пи",
    "Q": "кью",
    "R": "ар",
    "S": "эс",
    "T": "ти",
    "U": "ю",
    "V": "ви",
    "W": "дабл-ю",
    "X": "экс",
    "Y": "уай",
    "Z": "зэд",
}


_RU_UPPER = "А-ЯЁ"
_RU_LOWER = "а-яё"
_LETTER = "A-Za-zА-Яа-яЁё"
_TOKEN = rf"[{_LETTER}0-9]"
_GROUPED_INTEGER = r"(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)"
_SIGNED_NUMBER = rf"[−-]?{_GROUPED_INTEGER}(?:[.,]\d+)?"

_URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s]+")
_EMAIL_RE = re.compile(r"(?i)\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
_PATH_RE = re.compile(
    r"(?<!\w)(?:[A-Za-z]:[\\/]|/)(?:[^\s\\/]+[\\/])+[^\s]*"
)
_FILENAME_RE = re.compile(
    rf"(?<!\w)(?:[^\s/\\]+[/\\])*[^\s/\\]+\."
    rf"(?:[{_LETTER}]{{2,12}}|[chmr])(?![\w.])",
    re.IGNORECASE,
)
_SCIENTIFIC_E_RE = re.compile(
    r"(?<!\w)[−-]?\d+(?:[.,]\d+)?[eE][+−-]?\d+(?!\w)"
)
_SCIENTIFIC_POWER_RE = re.compile(
    r"(?<!\w)\d+(?:[.,]\d+)?\s*[×x]\s*10\^[−-]?\d+(?!\w)"
)
_EQUATION_RE = re.compile(
    rf"(?<!\w)(?=[{_LETTER}0-9().,^+*/=×\-\s]*=)"
    rf"[{_LETTER}0-9().,^]+(?:\s*[+*/=^×\-]\s*"
    rf"[{_LETTER}0-9().,^]+)+(?!\w)"
)
_MATH_ATOM = rf"(?:[{_LETTER}][{_LETTER}0-9_]*|\d+(?:[.,]\d+)?|\([^()\n]+\))"
_MATH_EXPRESSION_RE = re.compile(
    rf"(?<!\w){_MATH_ATOM}(?:\s*[+*/^×÷=<>≤≥≈≠]\s*{_MATH_ATOM})+"
    rf"(?!\w)"
)
_VARIABLE_CONTEXT_RE = re.compile(
    rf"(?i)\b(?:переменн(?:ая|ую|ой|ые|ых)|величин(?:а|у|ой|ы)|"
    rf"коэффициент|вектор|ось|точк(?:а|у|ой))\s+"
    rf"[{_RU_UPPER}A-Z][{_LETTER}0-9_]*(?!\w)"
)
_ID_TOKEN_RE = re.compile(
    rf"(?<!\w)(?=[{_LETTER}0-9_-]*\d)(?=[{_LETTER}0-9_-]*[{_LETTER}])"
    rf"{_TOKEN}+(?:[-_]{_TOKEN}+)*(?!\w)"
)
_UPPER_ID_RE = re.compile(
    r"(?<!\w)(?=[A-Z0-9_-]*[A-Z])[A-Z0-9]+(?:[-_][A-Z0-9]+)+(?!\w)"
)
_ID_CONTEXT_RE = re.compile(
    rf"(?i)\b(?:id|uid|uuid|guid|lab|идентификатор|лабораторн\w*)"
    rf"\s*[:#№-]?\s*[{_LETTER}0-9_-]+"
)
_RESERVED_ID_RE = re.compile(r"(?<!\w)(?:ID|UID|UUID|GUID|LAB)(?!\w)")

_SUBSCRIPT_DIGITS = "₀-₉"
_SUPERSCRIPT_DIGITS = "²³¹⁰⁴-⁹"
_SUPERSCRIPT_SIGNS = "⁺⁻"
_CHEMICAL_CANDIDATE_RE = re.compile(
    rf"(?<![A-Za-z0-9])(?:[A-Z][a-z]?|\d+|"
    rf"[{_SUBSCRIPT_DIGITS}{_SUPERSCRIPT_DIGITS}{_SUPERSCRIPT_SIGNS}"
    rf"()\[\]·.^+\-])+"
    rf"(?![A-Za-z0-9])"
)
_FORMULA_TOKEN = (
    rf"\d*[A-Z][A-Za-z0-9{_SUBSCRIPT_DIGITS}()\[\]·]*"
    rf"(?:[{_SUPERSCRIPT_DIGITS}]*[{_SUPERSCRIPT_SIGNS}])?"
)
_REACTION_RE = re.compile(
    rf"(?<![\w-]){_FORMULA_TOKEN}"
    rf"(?:\s*(?:<->|->|=>|[+→⟶↔⇄])\s*{_FORMULA_TOKEN})+(?![\w-])"
)
_ION_RE = re.compile(
    r"(?<![\w-])(?:[A-Z][a-z]?\d*)+\d*[+-]{1,2}(?![\w-])"
)
_ELEMENT_SYMBOLS = {
    "Ac", "Ag", "Al", "Am", "Ar", "As", "At", "Au", "B", "Ba", "Be",
    "Bh", "Bi", "Bk", "Br", "C", "Ca", "Cd", "Ce", "Cf", "Cl", "Cm",
    "Cn", "Co", "Cr", "Cs", "Cu", "Ds", "Dy", "Er", "Es", "Eu", "F",
    "Fe", "Fl", "Fm", "Fr", "Ga", "Gd", "Ge", "H", "He", "Hf", "Hg",
    "Ho", "Hs", "I", "In", "Ir", "K", "Kr", "La", "Li", "Lr", "Lu",
    "Lv", "Mc", "Md", "Mg", "Mn", "Mo", "Mt", "N", "Na", "Nb", "Nd",
    "Ne", "Nh", "Ni", "No", "Np", "O", "Og", "Os", "P", "Pa", "Pb",
    "Pd", "Pm", "Po", "Pr", "Pt", "Pu", "Ra", "Rb", "Re", "Rf", "Rg",
    "Rh", "Rn", "Ru", "S", "Sb", "Sc", "Se", "Sg", "Si", "Sm", "Sn",
    "Sr", "Ta", "Tb", "Tc", "Te", "Th", "Ti", "Tl", "Tm", "Ts", "U",
    "V", "W", "Xe", "Y", "Yb", "Zn", "Zr",
}
_ONE_LETTER_ELEMENTS = {symbol for symbol in _ELEMENT_SYMBOLS if len(symbol) == 1}

_EXPLICIT_ACRONYM_RE = re.compile(
    r"(?<![\w-])(?:API|DNA|AI|VR|pH)(?![\w-])"
)
_INITIALS_RE = re.compile(
    rf"(?<!\w)(?P<initials>(?:[{_RU_UPPER}A-Z]\.\s*){{1,3}})"
    rf"(?P<surname>[{_RU_UPPER}][{_RU_LOWER}]+"
    rf"(?:-[{_RU_UPPER}][{_RU_LOWER}]+)?)"
)
_LATIN_ACRONYM_RE = re.compile(r"(?<![\w-])[A-Z]{2,5}(?![\w-])")
_CYRILLIC_ACRONYM_RE = re.compile(
    rf"(?<![\w-])[{_RU_UPPER}]{{2,5}}(?![\w-])"
)
# В, С, К, О, У, А and И are ordinary Russian words, so a capital letter sitting
# in front of a lowercase word is a preposition and must not be spelled out.
# Label usage ("вариант А.", "точка Б)") is followed by punctuation and survives.
_INDIVIDUAL_CYRILLIC_RE = re.compile(
    rf"(?<!\w)(?P<letter>[{_RU_UPPER}])(?!\w)(?!\s+[{_RU_LOWER}])"
)
_NAMED_LATIN_LETTER_RE = re.compile(
    r"(?i)\b(?P<label>буква)\s+(?P<letter>[A-Z])(?!\w)"
)

_PHRASE_ABBREVIATIONS = (
    (re.compile(r"(?<!\w)и\s+т\s*\.\s*д\s*\.(?!\w)", re.IGNORECASE), "и так далее"),
    (
        re.compile(r"(?<!\w)и\s+т\s*\.\s*п\s*\.(?!\w)", re.IGNORECASE),
        "и тому подобное",
    ),
    (re.compile(r"(?<!\w)т\s*\.\s*е\s*\.(?!\w)", re.IGNORECASE), "то есть"),
    (re.compile(r"(?<!\w)т\s*\.\s*к\s*\.(?!\w)", re.IGNORECASE), "так как"),
)
_SHORT_ABBREVIATIONS = (
    (re.compile(r"(?<!\w)табл\.(?!\w)", re.IGNORECASE), "таблица"),
    (re.compile(r"(?<!\w)стр\.(?!\w)", re.IGNORECASE), "страница"),
    (re.compile(r"(?<!\w)рис\.(?!\w)", re.IGNORECASE), "рисунок"),
)
_CITY_ABBREVIATION_RE = re.compile(r"(?<!\w)г\.\s+(?=[А-ЯЁ])")
_YEAR_ABBREVIATION_RE = re.compile(r"(?<!\w)[гГ]\.(?!\w)")

_NUMBER_UNITS_HANDLED_LATER_RE = re.compile(
    rf"(?<!\w){_SIGNED_NUMBER}\s*(?:кг|мл|см)(?!\w)|"
    rf"(?<!\w){_SIGNED_NUMBER}\s*г(?![\w.])|"
    rf"(?<!\w){_SIGNED_NUMBER}\s*°\s*[CcСс](?!\w)",
    re.IGNORECASE,
)
_NUMBER_NEW_UNIT_RE = re.compile(
    rf"(?<!\w)(?P<number>{_SIGNED_NUMBER})\s*"
    r"(?P<unit>мг|км|мин|ч)(?!\w)",
    re.IGNORECASE,
)
_NUMBER_UNIT_FORMS = {
    "мг": ("миллиграмм", "миллиграмма", "миллиграммов"),
    "км": ("километр", "километра", "километров"),
    "мин": ("минута", "минуты", "минут"),
    "ч": ("час", "часа", "часов"),
}
_STANDALONE_DEGREE_RE = re.compile(r"(?<!\w)°\s*[CcСс](?!\w)")
_STANDALONE_UNIT_RE = re.compile(
    r"(?<![\w/])(?:мин|кг|мг|мл|км|см|ч|г)(?![\w/])"
)

_PROTECTOR_SLOTS = count()


@dataclass
class _Protector:
    values: list[str] = field(default_factory=list)
    # Protectors nest (the pipeline protects, then the abbreviation pass
    # protects again), so each instance needs its own placeholder namespace or
    # the inner restore overwrites spans the outer one still owns.
    # ponytail: 256 slots, plenty for the three protectors alive per call.
    slot: str = field(
        default_factory=lambda: chr(0xE200 + next(_PROTECTOR_SLOTS) % 0x100)
    )

    def _placeholder(self, index: int) -> str:
        return f"\ue000{self.slot}{chr(0xE300 + index)}\ue001"

    def protect(self, text: str, pattern: Pattern[str]) -> str:
        def replace(match: Match[str]) -> str:
            self.values.append(match.group(0))
            return self._placeholder(len(self.values) - 1)

        return pattern.sub(replace, text)

    def protect_value(self, value: str) -> str:
        self.values.append(value)
        return self._placeholder(len(self.values) - 1)

    def restore(self, text: str) -> str:
        for index, value in enumerate(self.values):
            text = text.replace(self._placeholder(index), value)
        return text


def _looks_like_chemical_formula(value: str) -> bool:
    # The candidate pattern swallows sentence punctuation, and speak_formula
    # puts it back, so a digitless formula must not fail on a trailing dot.
    value = value.strip(".")
    if any(char.isdigit() or char in "()[]·" for char in value):
        return bool(re.search(r"[A-Z]", value))

    symbols = re.findall(r"[A-Z][a-z]?", value)
    if not symbols or "".join(symbols) != value:
        return False
    if any(len(symbol) == 2 for symbol in symbols):
        return all(symbol in _ELEMENT_SYMBOLS for symbol in symbols)
    return len(symbols) > 1 and all(
        symbol in _ONE_LETTER_ELEMENTS for symbol in symbols
    )


def _protect_chemical_formulas(text: str, protector: _Protector) -> str:
    def replace(match: Match[str]) -> str:
        value = match.group(0)
        if not _looks_like_chemical_formula(value):
            return value
        return protector.protect_value(speak_formula(value))

    return _CHEMICAL_CANDIDATE_RE.sub(replace, text)


def _protect_reactions(text: str, protector: _Protector) -> str:
    def replace(match: Match[str]) -> str:
        value = match.group(0)
        spoken = speak_reaction(value)
        if spoken == value:
            return value
        return protector.protect_value(spoken)

    return _REACTION_RE.sub(replace, text)


def _protect_ions(text: str, protector: _Protector) -> str:
    def replace(match: Match[str]) -> str:
        value = match.group(0)
        spoken = speak_formula(value)
        if spoken == value:
            return value
        return protector.protect_value(spoken)

    return _ION_RE.sub(replace, text)


def _protect_nonlinguistic(text: str, protector: _Protector) -> str:
    for pattern in (
        _URL_RE,
        _EMAIL_RE,
        _PATH_RE,
        _FILENAME_RE,
        _SCIENTIFIC_E_RE,
        _SCIENTIFIC_POWER_RE,
    ):
        text = protector.protect(text, pattern)
    # Reactions must win over the equation and math patterns, which would
    # otherwise swallow the whole span and restore it unspoken.
    text = _protect_reactions(text, protector)
    # An ASCII charge sign lets the math pattern claim the ion, so ions have to
    # be spoken before the equation and math passes too.
    text = _protect_ions(text, protector)
    for pattern in (
        _EQUATION_RE,
        _MATH_EXPRESSION_RE,
        _VARIABLE_CONTEXT_RE,
        _ID_CONTEXT_RE,
    ):
        text = protector.protect(text, pattern)
    text = _protect_chemical_formulas(text, protector)
    for pattern in (
        _UPPER_ID_RE,
        _ID_TOKEN_RE,
        _RESERVED_ID_RE,
    ):
        text = protector.protect(text, pattern)
    return text


def _letter_name(letter: str) -> str:
    if letter in RUSSIAN_LETTER_NAMES:
        return RUSSIAN_LETTER_NAMES[letter]
    return LATIN_LETTER_NAMES_RU[letter]


def _replace_initials(match: Match[str]) -> str:
    letters = re.findall(rf"[{_RU_UPPER}A-Z]", match.group("initials"))
    spoken = " ".join(_letter_name(letter) for letter in letters)
    return f"{spoken} {match.group('surname')}"


def _unit_form_index(number_text: str) -> int:
    normalized = number_text.replace("−", "-")
    if "." in normalized or "," in normalized:
        return 1
    number = abs(int(re.sub(r"[ \u00a0\u202f]", "", normalized)))
    if number % 100 in range(11, 15):
        return 2
    if number % 10 == 1:
        return 0
    if number % 10 in range(2, 5):
        return 1
    return 2


def _replace_number_unit(match: Match[str]) -> str:
    unit = match.group("unit").lower()
    forms = _NUMBER_UNIT_FORMS[unit]
    return f"{match.group('number')} {forms[_unit_form_index(match.group('number'))]}"


def _replace_explicit_acronym(match: Match[str]) -> str:
    return RUSSIAN_ACRONYMS[match.group(0)]


def _replace_unknown_latin_acronym(match: Match[str]) -> str:
    value = match.group(0)
    return " ".join(LATIN_LETTER_NAMES_RU[letter] for letter in value)


def _replace_unknown_cyrillic_acronym(match: Match[str]) -> str:
    value = match.group(0)
    vowels = set("АЕЁИОУЫЭЮЯ")
    if any(letter in vowels for letter in value) and len(value) > 2:
        return value
    return " ".join(RUSSIAN_LETTER_NAMES[letter] for letter in value)


def normalize_russian_abbreviations(text: str) -> str:
    """Expand safe Russian abbreviations and letter sequences before TTS."""

    if not text:
        return text

    protector = _Protector()
    normalized = _protect_nonlinguistic(text, protector)

    normalized = _EXPLICIT_ACRONYM_RE.sub(_replace_explicit_acronym, normalized)
    normalized = _INITIALS_RE.sub(_replace_initials, normalized)

    for pattern, replacement in _PHRASE_ABBREVIATIONS:
        normalized = pattern.sub(replacement, normalized)
    for pattern, replacement in _SHORT_ABBREVIATIONS:
        normalized = pattern.sub(replacement, normalized)

    normalized = _CITY_ABBREVIATION_RE.sub("город ", normalized)
    normalized = _YEAR_ABBREVIATION_RE.sub("год", normalized)

    normalized = protector.protect(normalized, _NUMBER_UNITS_HANDLED_LATER_RE)
    normalized = _NUMBER_NEW_UNIT_RE.sub(_replace_number_unit, normalized)
    normalized = _STANDALONE_DEGREE_RE.sub("градус Цельсия", normalized)
    normalized = _STANDALONE_UNIT_RE.sub(
        lambda match: RUSSIAN_UNIT_ABBREVIATIONS[match.group(0)], normalized
    )

    normalized = _LATIN_ACRONYM_RE.sub(_replace_unknown_latin_acronym, normalized)
    normalized = _CYRILLIC_ACRONYM_RE.sub(
        _replace_unknown_cyrillic_acronym, normalized
    )
    normalized = _INDIVIDUAL_CYRILLIC_RE.sub(
        lambda match: RUSSIAN_LETTER_NAMES[match.group("letter")], normalized
    )
    normalized = _NAMED_LATIN_LETTER_RE.sub(
        lambda match: f"{match.group('label')} "
        f"{LATIN_LETTER_NAMES_RU[match.group('letter').upper()]}",
        normalized,
    )
    return protector.restore(normalized)


def normalize_russian_tts_text(text: str) -> str:
    """Run abbreviations before the unchanged Russian number normalizer."""

    from .text_normalization import normalize_russian_text

    protector = _Protector()
    protected = _protect_nonlinguistic(text, protector)
    normalized = normalize_russian_abbreviations(protected)
    normalized = normalize_russian_text(normalized)
    return protector.restore(normalized)


__all__ = [
    "LATIN_LETTER_NAMES_RU",
    "RUSSIAN_ABBREVIATIONS",
    "RUSSIAN_ACRONYMS",
    "RUSSIAN_LETTER_NAMES",
    "RUSSIAN_UNIT_ABBREVIATIONS",
    "normalize_russian_abbreviations",
    "normalize_russian_tts_text",
]
