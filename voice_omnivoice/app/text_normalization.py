"""Deterministic Kazakh number normalization for OmniVoice synthesis."""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Match, Pattern


_ONES = {
    0: "нөл",
    1: "бір",
    2: "екі",
    3: "үш",
    4: "төрт",
    5: "бес",
    6: "алты",
    7: "жеті",
    8: "сегіз",
    9: "тоғыз",
}
_TENS = {
    10: "он",
    20: "жиырма",
    30: "отыз",
    40: "қырық",
    50: "елу",
    60: "алпыс",
    70: "жетпіс",
    80: "сексен",
    90: "тоқсан",
}
_SCALES = (
    (1_000_000_000_000, "триллион"),
    (1_000_000_000, "миллиард"),
    (1_000_000, "миллион"),
    (1_000, "мың"),
)
_ORDINAL_LAST_WORD = {
    "нөл": "нөлінші",
    "бір": "бірінші",
    "екі": "екінші",
    "үш": "үшінші",
    "төрт": "төртінші",
    "бес": "бесінші",
    "алты": "алтыншы",
    "жеті": "жетінші",
    "сегіз": "сегізінші",
    "тоғыз": "тоғызыншы",
    "он": "оныншы",
    "жиырма": "жиырмасыншы",
    "отыз": "отызыншы",
    "қырық": "қырқыншы",
    "елу": "елуінші",
    "алпыс": "алпысыншы",
    "жетпіс": "жетпісінші",
    "сексен": "сексенінші",
    "тоқсан": "тоқсаныншы",
    "жүз": "жүзінші",
    "мың": "мыңыншы",
    "миллион": "миллионыншы",
    "миллиард": "миллиардыншы",
    "триллион": "триллионыншы",
}
_MONTHS = {
    1: "қаңтар",
    2: "ақпан",
    3: "наурыз",
    4: "сәуір",
    5: "мамыр",
    6: "маусым",
    7: "шілде",
    8: "тамыз",
    9: "қыркүйек",
    10: "қазан",
    11: "қараша",
    12: "желтоқсан",
}

_LETTER = "A-Za-zА-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі"
_GROUPED_INTEGER = r"(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)"
_SIGNED_INTEGER = rf"[−-]?{_GROUPED_INTEGER}"
_SIGNED_NUMBER = rf"{_SIGNED_INTEGER}(?:[.,]\d+)?"

_URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s]+")
_EMAIL_RE = re.compile(r"(?i)\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
_FILENAME_RE = re.compile(
    rf"(?<!\w)(?:[^\s/\\]+[/\\])*[^\s/\\]+\.[{_LETTER}]{{1,12}}(?!\w)"
)
_SCIENTIFIC_E_RE = re.compile(
    r"(?<!\w)[−-]?\d+(?:[.,]\d+)?[eE][+−-]?\d+(?!\w)"
)
_SCIENTIFIC_POWER_RE = re.compile(
    r"(?<!\w)\d+(?:[.,]\d+)?\s*[×x]\s*10\^[−-]?\d+(?!\w)"
)
_EQUATION_RE = re.compile(
    rf"(?<!\w)(?=[{_LETTER}0-9().,^+*/=×\-\s]*=)"
    rf"[{_LETTER}0-9().,^]+(?:\s*[+*/=^×\-]\s*[{_LETTER}0-9().,^]+)+(?!\w)"
)
_CHEMICAL_FORMULA_RE = re.compile(
    r"(?<![A-Za-z0-9])(?=[A-Za-z0-9()\[\]·.^+\-]*[A-Z])"
    r"(?=[A-Za-z0-9()\[\]·.^+\-]*?(?:\d|"
    r"[A-Z][a-z]?[A-Za-z0-9()\[\]·.^+\-]*[A-Z]))"
    r"(?:[A-Z][a-z]?|[0-9()\[\]·.^+\-])+(?![A-Za-z0-9])"
)
_ADJACENT_IDENTIFIER_RE = re.compile(
    rf"(?<!\w)(?=[{_LETTER}0-9]*[{_LETTER}])(?=[{_LETTER}0-9]*\d)"
    rf"[{_LETTER}0-9]+(?!\w)"
)
_SEPARATED_IDENTIFIER_RE = re.compile(
    rf"(?<!\w)(?=[{_LETTER}0-9_-]*\d)[{_LETTER}][{_LETTER}0-9]*"
    rf"(?:[-_][{_LETTER}0-9]+)+(?!\w)"
)
_NUMERIC_IDENTIFIER_RE = re.compile(r"(?<!\w)\d+(?:-\d+){2,}(?!\w)")
_DOTTED_IDENTIFIER_RE = re.compile(r"(?<!\w)\d+(?:\.\d+){2,}(?!\w)")
_TIME_LIKE_RE = re.compile(r"(?<!\d)\d{1,2}:\d{2}(?:-(?:да|де|та|те))?(?!\d)", re.IGNORECASE)

_DATE_RE = re.compile(
    r"(?<![\w.])(?P<day>\d{1,2})\.(?P<month>\d{1,2})\.(?P<year>\d{4})(?![\w.])"
)
_CLOCK_WITH_WORD_RE = re.compile(
    r"\bсағат\s+(?P<hour>\d{1,2}):(?P<minute>\d{2})(?P<case>-(?:да|де|та|те))?",
    re.IGNORECASE,
)
_TIME_RE = re.compile(
    r"(?<![\d:])(?P<hour>\d{1,2}):(?P<minute>\d{2})(?P<case>-(?:да|де|та|те))?(?![\d:])",
    re.IGNORECASE,
)
_RANGE_RE = re.compile(
    rf"(?<![\w.,])(?P<start>{_GROUPED_INTEGER})\s*[–-]\s*"
    rf"(?P<end>{_GROUPED_INTEGER})(?!\w|[.,]\d)"
)
_TEMPERATURE_RE = re.compile(
    rf"(?<!\w)(?P<number>{_SIGNED_NUMBER})\s*°\s*[CcСс]"
    r"(?P<case>-(?:қа|ке|ға|ге))?(?!\w)",
    re.IGNORECASE,
)
_UNIT_RE = re.compile(
    rf"(?<!\w)(?P<number>{_SIGNED_NUMBER})\s*"
    r"(?P<unit>мл|кг|см|мм|минут(?:ы|тің|та|те|тан|тен)?|секунд(?:ы|тың|та|те|тан|тен)?|"
    r"сағат(?:ы|тың|та|те|тан|тен)?|л|г|м)(?!\w)",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(
    rf"(?<!\w)(?P<number>{_SIGNED_NUMBER})\s*%(?!\w)"
)
_DECIMAL_RE = re.compile(
    rf"(?<![\w.,])(?P<number>{_SIGNED_INTEGER}[.,]\d+)(?!\w|[.,]\d)"
)
_INTEGER_RE = re.compile(
    rf"(?<![\w.,])(?P<number>{_SIGNED_INTEGER})(?!\w|[.,]\d)"
)

_UNIT_NAMES = {
    "мл": "миллилитр",
    "л": "литр",
    "г": "грамм",
    "кг": "килограмм",
    "м": "метр",
    "см": "сантиметр",
    "мм": "миллиметр",
    "минут": "минут",
    "секунд": "секунд",
    "сағат": "сағат",
}


# Placeholders from a nested protector must not collide with the ones minted
# by the abbreviation normalizer, so each instance takes its own slot char.
_SLOTS = itertools.count()


def _next_slot() -> str:
    return chr(0xE090 + next(_SLOTS) % 0x70)


@dataclass
class _Protector:
    values: list[str] = field(default_factory=list)
    slot: str = field(default_factory=_next_slot)

    def _placeholder(self, index: int) -> str:
        return f"\ue000{self.slot}{chr(0xE100 + index)}\ue001"

    def protect(self, text: str, pattern: Pattern[str]) -> str:
        def replace(match: Match[str]) -> str:
            index = len(self.values)
            self.values.append(match.group(0))
            return self._placeholder(index)

        return pattern.sub(replace, text)

    def restore(self, text: str) -> str:
        for index, value in enumerate(self.values):
            text = text.replace(self._placeholder(index), value)
        return text


def _under_thousand(number: int) -> list[str]:
    words: list[str] = []
    hundreds, remainder = divmod(number, 100)
    if hundreds:
        if hundreds > 1:
            words.append(_ONES[hundreds])
        words.append("жүз")
    if remainder >= 10:
        tens, ones = divmod(remainder, 10)
        words.append(_TENS[tens * 10])
        if ones:
            words.append(_ONES[ones])
    elif remainder:
        words.append(_ONES[remainder])
    return words


def kazakh_cardinal(number: int) -> str:
    """Return a Kazakh cardinal for an integer with deterministic spacing."""

    if number == 0:
        return _ONES[0]
    if number < 0:
        return f"минус {kazakh_cardinal(-number)}"

    words: list[str] = []
    remainder = number
    for scale_value, scale_word in _SCALES:
        group, remainder = divmod(remainder, scale_value)
        if group:
            words.extend(_integer_words(group))
            words.append(scale_word)
    words.extend(_under_thousand(remainder))
    return " ".join(words)


def _integer_words(number: int) -> list[str]:
    return kazakh_cardinal(number).split()


def kazakh_ordinal(number: int) -> str:
    """Return the ordinal form used for spoken dates."""

    cardinal = kazakh_cardinal(number)
    words = cardinal.split()
    last = words[-1]
    words[-1] = _ORDINAL_LAST_WORD[last]
    return " ".join(words)


def _parse_integer(value: str) -> int:
    compact = re.sub(r"[ \u00a0\u202f]", "", value)
    return int(compact.replace("−", "-"))


def _number_words(value: str) -> str:
    normalized = value.replace("−", "-")
    sign = ""
    if normalized.startswith("-"):
        sign = "минус "
        normalized = normalized[1:]

    if "." not in normalized and "," not in normalized:
        return sign + kazakh_cardinal(_parse_integer(normalized))

    integer_part, fraction_part = re.split(r"[.,]", normalized, maxsplit=1)
    denominator = 10 ** len(fraction_part)
    denominator_words = _ablative(kazakh_cardinal(denominator))
    numerator = int(fraction_part)
    return (
        f"{sign}{kazakh_cardinal(_parse_integer(integer_part))} бүтін "
        f"{denominator_words} {kazakh_cardinal(numerator)}"
    )


def _front_vowel(word: str) -> bool:
    vowels = [char for char in word.lower() if char in "аәеёиіоөұүуыэюя"]
    return bool(vowels) and vowels[-1] in "әеіөүэ"


def _append_to_last_word(phrase: str, suffix: str) -> str:
    words = phrase.split()
    words[-1] += suffix
    return " ".join(words)


def _ablative(phrase: str) -> str:
    last = phrase[-1].lower()
    front = _front_vowel(phrase)
    if last in "лмнң":
        suffix = "нен" if front else "нан"
    elif last in "пфкқтсшщчцхһ":
        suffix = "тен" if front else "тан"
    else:
        suffix = "ден" if front else "дан"
    return _append_to_last_word(phrase, suffix)


def _dative(phrase: str) -> str:
    last = phrase[-1].lower()
    front = _front_vowel(phrase)
    if last in "пфкқтсшщчцхһ":
        suffix = "ке" if front else "қа"
    else:
        suffix = "ге" if front else "ға"
    return _append_to_last_word(phrase, suffix)


def _locative(phrase: str) -> str:
    last = phrase[-1].lower()
    front = _front_vowel(phrase)
    if last in "пфкқтсшщчцхһ":
        suffix = "те" if front else "та"
    else:
        suffix = "де" if front else "да"
    return _append_to_last_word(phrase, suffix)


def _mask_nonlinguistic(text: str, protector: _Protector) -> str:
    for pattern in (
        _URL_RE,
        _EMAIL_RE,
        _FILENAME_RE,
        _SCIENTIFIC_E_RE,
        _SCIENTIFIC_POWER_RE,
        _EQUATION_RE,
        _CHEMICAL_FORMULA_RE,
        _SEPARATED_IDENTIFIER_RE,
        _ADJACENT_IDENTIFIER_RE,
        _NUMERIC_IDENTIFIER_RE,
    ):
        text = protector.protect(text, pattern)
    return text


def _replace_date(match: Match[str]) -> str:
    day = int(match.group("day"))
    month = int(match.group("month"))
    year = int(match.group("year"))
    try:
        date(year, month, day)
    except ValueError:
        return match.group(0)
    return f"{kazakh_ordinal(year)} жылғы {kazakh_ordinal(day)} {_MONTHS[month]}"


def _valid_time(match: Match[str]) -> tuple[int, int] | None:
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    if hour > 23 or minute > 59:
        return None
    return hour, minute


def _digital_minute_words(minute: int) -> str:
    if minute < 10:
        return f"нөл {kazakh_cardinal(minute)}"
    return kazakh_cardinal(minute)


def _replace_clock_with_word(match: Match[str]) -> str:
    parsed = _valid_time(match)
    if parsed is None:
        return match.group(0)
    hour, minute = parsed
    clock = f"сағат {kazakh_cardinal(hour)} {_digital_minute_words(minute)}"
    return _locative(clock) if match.group("case") else clock


def _replace_time(match: Match[str]) -> str:
    parsed = _valid_time(match)
    if parsed is None:
        return match.group(0)
    hour, minute = parsed
    phrase = (
        f"{kazakh_cardinal(hour)} сағат {kazakh_cardinal(minute)} минут"
    )
    return _locative(phrase) if match.group("case") else phrase


def _replace_range(match: Match[str]) -> str:
    start = kazakh_cardinal(_parse_integer(match.group("start")))
    end = kazakh_cardinal(_parse_integer(match.group("end")))
    return f"{_ablative(start)} {_dative(end)} дейін"


def _replace_temperature(match: Match[str]) -> str:
    phrase = f"{_number_words(match.group('number'))} градус Цельсий"
    if match.group("case"):
        return f"{_number_words(match.group('number'))} градус Цельсийге"
    return phrase


def _replace_unit(match: Match[str]) -> str:
    unit = match.group("unit").lower()
    canonical = max(
        (
            (prefix, name)
            for prefix, name in _UNIT_NAMES.items()
            if unit.startswith(prefix)
        ),
        key=lambda item: len(item[0]),
    )[1]
    return f"{_number_words(match.group('number'))} {canonical}"


def normalize_kazakh_text(text: str) -> str:
    """Expand supported numeric forms while preserving non-linguistic tokens."""

    if not text or not any(char.isdigit() for char in text):
        return text

    protector = _Protector()
    normalized = _mask_nonlinguistic(text, protector)
    normalized = _DATE_RE.sub(_replace_date, normalized)
    normalized = protector.protect(normalized, _DOTTED_IDENTIFIER_RE)
    normalized = _CLOCK_WITH_WORD_RE.sub(_replace_clock_with_word, normalized)
    normalized = _TIME_RE.sub(_replace_time, normalized)
    normalized = protector.protect(normalized, _TIME_LIKE_RE)
    normalized = _RANGE_RE.sub(_replace_range, normalized)
    normalized = _TEMPERATURE_RE.sub(_replace_temperature, normalized)
    normalized = _UNIT_RE.sub(_replace_unit, normalized)
    normalized = _PERCENT_RE.sub(
        lambda match: f"{_number_words(match.group('number'))} пайыз", normalized
    )
    normalized = _DECIMAL_RE.sub(
        lambda match: _number_words(match.group("number")), normalized
    )
    normalized = _INTEGER_RE.sub(
        lambda match: kazakh_cardinal(_parse_integer(match.group("number"))),
        normalized,
    )
    return protector.restore(normalized)


__all__ = ["kazakh_cardinal", "kazakh_ordinal", "normalize_kazakh_text"]
