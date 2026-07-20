"""Kazakh abbreviation, initial, and letter normalization for TTS."""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import Match, Pattern

from .formula_speech import speak_formula, speak_reaction


KAZAKH_ABBREVIATIONS = {
    "т.б.": "тағы басқа",
    "т.с.с.": "тағы сол сияқты",
    "т.с.": "тағы сол",
    "ж.": "жыл",
    "б.": "бет",
}

KAZAKH_UNIT_ABBREVIATIONS = {
    "кг": "килограмм",
    "г": "грамм",
    "мг": "миллиграмм",
    "мл": "миллилитр",
    "л": "литр",
    "мм": "миллиметр",
    "см": "сантиметр",
    "м": "метр",
    "км": "километр",
    "с": "секунд",
    "мин": "минут",
    "сағ": "сағат",
    "Н": "ньютон",
    "Дж": "джоуль",
    "Вт": "ватт",
    "кВт": "киловатт",
    "Па": "паскаль",
    "кПа": "килопаскаль",
    "Гц": "герц",
    "°C": "градус Цельсий",
    "м/с": "секундына метр",
    "км/сағ": "сағатына километр",
}

KAZAKH_ACRONYMS = {
    "AI": "эй ай",
    "VR": "ви ар",
    "API": "эй пи ай",
    "pH": "пэ аш",
    "DNA": "ди эн эй",
}

KAZAKH_LETTER_NAMES = {
    "А": "а",
    "Ә": "ә",
    "Б": "бэ",
    "В": "вэ",
    "Г": "гэ",
    "Ғ": "ғы",
    "Д": "дэ",
    "Е": "е",
    "Ё": "ё",
    "Ж": "жэ",
    "З": "зэ",
    "И": "и",
    "Й": "қысқа и",
    "К": "ка",
    "Қ": "қы",
    "Л": "эл",
    "М": "эм",
    "Н": "эн",
    "Ң": "ең",
    "О": "о",
    "Ө": "ө",
    "П": "пэ",
    "Р": "эр",
    "С": "эс",
    "Т": "тэ",
    "У": "у",
    "Ұ": "ұ",
    "Ү": "ү",
    "Ф": "эф",
    "Х": "ха",
    "Һ": "һа",
    "Ц": "цэ",
    "Ч": "че",
    "Ш": "ша",
    "Щ": "ща",
    "Ъ": "айыру белгісі",
    "Ы": "ы",
    "І": "і",
    "Ь": "жіңішкелік белгісі",
    "Э": "э",
    "Ю": "ю",
    "Я": "я",
}

LATIN_LETTER_NAMES_KK = {
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


_KK_UPPER = "А-ЯЁӘҒҚҢӨҰҮҺІ"
_KK_LOWER = "а-яёәғқңөұүһі"
_LETTER = "A-Za-zА-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі"
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
    rf"(?i)\b(?:айнымалы|шама|коэффициент|вектор|ось|нүкте)"
    rf"\s+[{_KK_UPPER}A-Z][{_LETTER}0-9_]*(?!\w)"
)
_ID_TOKEN_RE = re.compile(
    rf"(?<!\w)(?=[{_LETTER}0-9_-]*\d)(?=[{_LETTER}0-9_-]*[{_LETTER}])"
    rf"{_TOKEN}+(?:[-_]{_TOKEN}+)*(?!\w)"
)
_UPPER_ID_RE = re.compile(
    r"(?<!\w)(?=[A-Z0-9_-]*[A-Z])[A-Z0-9]+(?:[-_][A-Z0-9]+)+(?!\w)"
)
_ID_CONTEXT_RE = re.compile(
    rf"(?i)\b(?:id|uid|uuid|guid|lab|идентификатор|зертхана\w*)"
    rf"\s*[:#№-]?\s*[{_LETTER}0-9_-]+"
)
_RESERVED_ID_RE = re.compile(r"(?<!\w)(?:ID|UID|UUID|GUID|LAB)(?!\w)")

_SUB_SUPERSCRIPTS = "₀-₉⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻"
_CHEMICAL_CANDIDATE_RE = re.compile(
    rf"(?<![A-Za-z0-9])(?:[A-Z][a-z]?|\d+|[{_SUB_SUPERSCRIPTS}()\[\]·.^+\-])+"
    r"(?![A-Za-z0-9])"
)
_ARROW = r"(?:->|=>|→|⟶|↔|⇄|⇌|⇔)"
_REACTION_PART = rf"[A-Za-z0-9()\[\]·{_SUB_SUPERSCRIPTS}]+"
_REACTION_RE = re.compile(
    rf"(?<![\w-]){_REACTION_PART}"
    rf"(?:\s*(?:\+|{_ARROW})\s*{_REACTION_PART})+(?![\w-])"
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
    rf"(?<!\w)(?P<initials>(?:[{_KK_UPPER}A-Z]\.\s*){{1,3}})"
    rf"(?P<surname>[{_KK_UPPER}][{_KK_LOWER}]+"
    rf"(?:-[{_KK_UPPER}][{_KK_LOWER}]+)?)"
)
_LATIN_ACRONYM_RE = re.compile(r"(?<![\w-])[A-Z]{2,5}(?![\w-])")
_CYRILLIC_ACRONYM_RE = re.compile(
    rf"(?<![\w-])[{_KK_UPPER}]{{2,5}}(?![\w-])"
)
# A capital letter sitting in front of a lowercase word is a preposition such as
# В or О, not a letter to spell out. Label usage ("А нұсқасы.") keeps its
# punctuation and still gets spelled.
_INDIVIDUAL_CYRILLIC_RE = re.compile(
    rf"(?<!\w)(?P<letter>[{_KK_UPPER}])(?!\w)(?!\s+[{_KK_LOWER}])"
)
_NAMED_LATIN_LETTER_RE = re.compile(
    r"(?i)\b(?P<label>әріп|әрпі)\s+(?P<letter>[A-Z])(?!\w)"
)

_PHRASE_ABBREVIATIONS = (
    (
        re.compile(
            r"(?<!\w)т\s*\.\s*с\s*\.\s*с\s*\.(?!\w)", re.IGNORECASE
        ),
        "тағы сол сияқты",
    ),
    (re.compile(r"(?<!\w)т\s*\.\s*б\s*\.(?!\w)", re.IGNORECASE), "тағы басқа"),
    (re.compile(r"(?<!\w)т\s*\.\s*с\s*\.(?!\w)", re.IGNORECASE), "тағы сол"),
)
_SHORT_ABBREVIATIONS = (
    (re.compile(r"(?<!\w)[жЖ]\.(?!\w)"), "жыл"),
    (re.compile(r"(?<!\w)[бБ]\.(?!\w)"), "бет"),
)

_NUMBER_UNITS_HANDLED_LATER_RE = re.compile(
    rf"(?<!\w){_SIGNED_NUMBER}\s*(?:кг|мл|см|мм|л|г|м)(?![\w/])|"
    rf"(?<!\w){_SIGNED_NUMBER}\s*°\s*[CcСс]"
    r"(?:-(?:қа|ке|ға|ге))?(?!\w)",
    re.IGNORECASE,
)
_NUMBER_COMPOUND_UNIT_RE = re.compile(
    rf"(?<!\w)(?P<number>{_SIGNED_NUMBER})\s*"
    r"(?P<unit>км/сағ|м/с)(?!\w)"
)
_NUMBER_NEW_UNIT_RE = re.compile(
    rf"(?<!\w)(?P<number>{_SIGNED_NUMBER})\s*"
    r"(?P<unit>кПа|кВт|мг|км|мин|сағ|Дж|Вт|Па|Гц|Н|с)(?!\w)"
)
_NUMBER_COMPOUND_UNIT_TEMPLATES = {
    "м/с": "секундына {number} метр",
    "км/сағ": "сағатына {number} километр",
}
_STANDALONE_DEGREE_RE = re.compile(r"(?<!\w)°\s*[CcСс](?!\w)")
_STANDALONE_COMPOUND_UNIT_RE = re.compile(
    r"(?<!\w)(?:км/сағ|м/с)(?!\w)"
)
_STANDALONE_UNIT_RE = re.compile(
    r"(?<![\w/])(?:кПа|кВт|сағ|мин|кг|мг|мл|км|см|мм|"
    r"Дж|Вт|Па|Гц|Н|с|г|л|м)"
    r"(?![\w/])"
)


# Protectors nest (this module runs inside the number normalizer), so each
# instance gets its own placeholder slot: a shared one lets the inner restore
# overwrite the outer instance's spans.
_SLOTS = itertools.count()


def _next_slot() -> str:
    return chr(0xE010 + next(_SLOTS) % 0x80)


@dataclass
class _Protector:
    values: list[str] = field(default_factory=list)
    slot: str = field(default_factory=_next_slot)

    def _placeholder(self, index: int) -> str:
        return f"\ue000{self.slot}{chr(0xE100 + index)}\ue001"

    def protect(self, text: str, pattern: Pattern[str]) -> str:
        def replace(match: Match[str]) -> str:
            return self.protect_value(match.group(0))

        return pattern.sub(replace, text)

    def protect_value(self, value: str) -> str:
        index = len(self.values)
        self.values.append(value)
        return self._placeholder(index)

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
        spoken = speak_formula(value)
        if spoken == value:
            # Acronyms such as CU or ISBN: let the acronym handler spell them.
            return value
        return protector.protect_value(spoken)

    return _CHEMICAL_CANDIDATE_RE.sub(replace, text)


def _protect_reactions(text: str, protector: _Protector) -> str:
    def replace(match: Match[str]) -> str:
        value = match.group(0)
        spoken = speak_reaction(value)
        if spoken == value:
            # Not chemistry, leave the span to the math protection below.
            return value
        return protector.protect_value(spoken)

    return _REACTION_RE.sub(replace, text)


def _protect_ions(text: str, protector: _Protector) -> str:
    def replace(match: Match[str]) -> str:
        value = match.group(0)
        spoken = speak_formula(value)
        if spoken == value:
            # Not chemistry, leave the span to the math protection below.
            return value
        return protector.protect_value(spoken)

    return _ION_RE.sub(replace, text)


def _protect_math_expressions(text: str, protector: _Protector) -> str:
    def replace(match: Match[str]) -> str:
        value = match.group(0)
        if re.sub(r"\s+", "", value) in {"м/с", "км/сағ"}:
            return value
        return protector.protect_value(value)

    return _MATH_EXPRESSION_RE.sub(replace, text)


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
    text = _protect_reactions(text, protector)
    # An ASCII charge sign lets the math pattern claim the ion, so ions have to
    # be spoken before the equation and math passes too.
    text = _protect_ions(text, protector)
    text = protector.protect(text, _EQUATION_RE)
    text = _protect_math_expressions(text, protector)
    for pattern in (
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
    if letter in KAZAKH_LETTER_NAMES:
        return KAZAKH_LETTER_NAMES[letter]
    return LATIN_LETTER_NAMES_KK[letter]


def _replace_initials(match: Match[str]) -> str:
    letters = re.findall(rf"[{_KK_UPPER}A-Z]", match.group("initials"))
    spoken = " ".join(_letter_name(letter) for letter in letters)
    return f"{spoken} {match.group('surname')}"


def _replace_explicit_acronym(match: Match[str]) -> str:
    return KAZAKH_ACRONYMS[match.group(0)]


def _replace_unknown_latin_acronym(match: Match[str]) -> str:
    value = match.group(0)
    return " ".join(LATIN_LETTER_NAMES_KK[letter] for letter in value)


def _replace_unknown_cyrillic_acronym(match: Match[str]) -> str:
    value = match.group(0)
    vowels = set("АӘЕЁИІОӨҰҮУЫЭЮЯ")
    if any(letter in vowels for letter in value) and len(value) > 2:
        return value
    return " ".join(KAZAKH_LETTER_NAMES[letter] for letter in value)


def _replace_number_compound_unit(match: Match[str]) -> str:
    return _NUMBER_COMPOUND_UNIT_TEMPLATES[match.group("unit")].format(
        number=match.group("number")
    )


def normalize_kazakh_abbreviations(text: str) -> str:
    """Expand safe Kazakh abbreviations and letter sequences before TTS."""

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

    normalized = protector.protect(normalized, _NUMBER_UNITS_HANDLED_LATER_RE)
    normalized = _NUMBER_COMPOUND_UNIT_RE.sub(
        _replace_number_compound_unit, normalized
    )
    normalized = _NUMBER_NEW_UNIT_RE.sub(
        lambda match: f"{match.group('number')} "
        f"{KAZAKH_UNIT_ABBREVIATIONS[match.group('unit')]}",
        normalized,
    )
    normalized = _STANDALONE_DEGREE_RE.sub("градус Цельсий", normalized)
    normalized = _STANDALONE_COMPOUND_UNIT_RE.sub(
        lambda match: KAZAKH_UNIT_ABBREVIATIONS[match.group(0)], normalized
    )
    normalized = _STANDALONE_UNIT_RE.sub(
        lambda match: KAZAKH_UNIT_ABBREVIATIONS[match.group(0)], normalized
    )

    normalized = _LATIN_ACRONYM_RE.sub(_replace_unknown_latin_acronym, normalized)
    normalized = _CYRILLIC_ACRONYM_RE.sub(
        _replace_unknown_cyrillic_acronym, normalized
    )
    normalized = _INDIVIDUAL_CYRILLIC_RE.sub(
        lambda match: KAZAKH_LETTER_NAMES[match.group("letter")], normalized
    )
    normalized = _NAMED_LATIN_LETTER_RE.sub(
        lambda match: f"{match.group('label')} "
        f"{LATIN_LETTER_NAMES_KK[match.group('letter').upper()]}",
        normalized,
    )
    return protector.restore(normalized)


def normalize_kazakh_tts_text(text: str) -> str:
    """Run abbreviations before the unchanged Kazakh number normalizer."""

    from .text_normalization import normalize_kazakh_text

    protector = _Protector()
    protected = _protect_nonlinguistic(text, protector)
    normalized = normalize_kazakh_abbreviations(protected)
    normalized = normalize_kazakh_text(normalized)
    return protector.restore(normalized)


__all__ = [
    "KAZAKH_ABBREVIATIONS",
    "KAZAKH_ACRONYMS",
    "KAZAKH_LETTER_NAMES",
    "KAZAKH_UNIT_ABBREVIATIONS",
    "LATIN_LETTER_NAMES_KK",
    "normalize_kazakh_abbreviations",
    "normalize_kazakh_tts_text",
]
