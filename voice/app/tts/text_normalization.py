"""Deterministic Russian number normalization for local TTS backends."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Match, Pattern, TypeVar


_T = TypeVar("_T")

_ONES = {
    "masculine": {
        0: "ноль",
        1: "один",
        2: "два",
        3: "три",
        4: "четыре",
        5: "пять",
        6: "шесть",
        7: "семь",
        8: "восемь",
        9: "девять",
    },
    "feminine": {
        0: "ноль",
        1: "одна",
        2: "две",
        3: "три",
        4: "четыре",
        5: "пять",
        6: "шесть",
        7: "семь",
        8: "восемь",
        9: "девять",
    },
}
_TEENS = {
    10: "десять",
    11: "одиннадцать",
    12: "двенадцать",
    13: "тринадцать",
    14: "четырнадцать",
    15: "пятнадцать",
    16: "шестнадцать",
    17: "семнадцать",
    18: "восемнадцать",
    19: "девятнадцать",
}
_TENS = {
    20: "двадцать",
    30: "тридцать",
    40: "сорок",
    50: "пятьдесят",
    60: "шестьдесят",
    70: "семьдесят",
    80: "восемьдесят",
    90: "девяносто",
}
_HUNDREDS = {
    100: "сто",
    200: "двести",
    300: "триста",
    400: "четыреста",
    500: "пятьсот",
    600: "шестьсот",
    700: "семьсот",
    800: "восемьсот",
    900: "девятьсот",
}
_SCALES = (
    (1_000_000_000_000, ("триллион", "триллиона", "триллионов"), "masculine"),
    (1_000_000_000, ("миллиард", "миллиарда", "миллиардов"), "masculine"),
    (1_000_000, ("миллион", "миллиона", "миллионов"), "masculine"),
    (1_000, ("тысяча", "тысячи", "тысяч"), "feminine"),
)
_GENITIVE_ONES = {
    "masculine": {
        0: "нуля",
        1: "одного",
        2: "двух",
        3: "трёх",
        4: "четырёх",
        5: "пяти",
        6: "шести",
        7: "семи",
        8: "восьми",
        9: "девяти",
    },
    "feminine": {
        0: "нуля",
        1: "одной",
        2: "двух",
        3: "трёх",
        4: "четырёх",
        5: "пяти",
        6: "шести",
        7: "семи",
        8: "восьми",
        9: "девяти",
    },
}
_GENITIVE_TEENS = {
    10: "десяти",
    11: "одиннадцати",
    12: "двенадцати",
    13: "тринадцати",
    14: "четырнадцати",
    15: "пятнадцати",
    16: "шестнадцати",
    17: "семнадцати",
    18: "восемнадцати",
    19: "девятнадцати",
}
_GENITIVE_TENS = {
    20: "двадцати",
    30: "тридцати",
    40: "сорока",
    50: "пятидесяти",
    60: "шестидесяти",
    70: "семидесяти",
    80: "восьмидесяти",
    90: "девяноста",
}
_GENITIVE_HUNDREDS = {
    100: "ста",
    200: "двухсот",
    300: "трёхсот",
    400: "четырёхсот",
    500: "пятисот",
    600: "шестисот",
    700: "семисот",
    800: "восьмисот",
    900: "девятисот",
}
_ORDINAL_GENITIVE = {
    1: "первого",
    2: "второго",
    3: "третьего",
    4: "четвёртого",
    5: "пятого",
    6: "шестого",
    7: "седьмого",
    8: "восьмого",
    9: "девятого",
    10: "десятого",
    11: "одиннадцатого",
    12: "двенадцатого",
    13: "тринадцатого",
    14: "четырнадцатого",
    15: "пятнадцатого",
    16: "шестнадцатого",
    17: "семнадцатого",
    18: "восемнадцатого",
    19: "девятнадцатого",
    20: "двадцатого",
    30: "тридцатого",
    40: "сорокового",
    50: "пятидесятого",
    60: "шестидесятого",
    70: "семидесятого",
    80: "восьмидесятого",
    90: "девяностого",
    100: "сотого",
    200: "двухсотого",
    300: "трёхсотого",
    400: "четырёхсотого",
    500: "пятисотого",
    600: "шестисотого",
    700: "семисотого",
    800: "восьмисотого",
    900: "девятисотого",
    1_000: "тысячного",
    2_000: "двухтысячного",
}
_MONTHS_GENITIVE = {
    1: "января",
    2: "февраля",
    3: "марта",
    4: "апреля",
    5: "мая",
    6: "июня",
    7: "июля",
    8: "августа",
    9: "сентября",
    10: "октября",
    11: "ноября",
    12: "декабря",
}
_DECIMAL_DENOMINATORS = {
    1: ("десятая", "десятых"),
    2: ("сотая", "сотых"),
    3: ("тысячная", "тысячных"),
    4: ("десятитысячная", "десятитысячных"),
    5: ("стотысячная", "стотысячных"),
    6: ("миллионная", "миллионных"),
}

_LETTER = "A-Za-zА-Яа-яЁё"
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
_TIME_LIKE_RE = re.compile(r"(?<!\d)\d{1,2}:\d{2}(?!\d)")

_DATE_RE = re.compile(
    r"(?<![\w.])(?P<day>\d{1,2})\.(?P<month>\d{1,2})\.(?P<year>\d{4})(?![\w.])"
)
_TIME_RE = re.compile(
    r"(?<![\d:])(?P<hour>\d{1,2}):(?P<minute>\d{2})(?![\d:])"
)
_RANGE_RE = re.compile(
    rf"(?<![\w.,])(?P<start>{_GROUPED_INTEGER})\s*[–-]\s*"
    rf"(?P<end>{_GROUPED_INTEGER})(?!\w|[.,]\d)"
)
_TEMPERATURE_AFTER_DO_RE = re.compile(
    rf"(?P<preposition>\bдо\s+)(?P<number>{_SIGNED_INTEGER})\s*°\s*[CcСс](?!\w)",
    re.IGNORECASE,
)
_TEMPERATURE_RE = re.compile(
    rf"(?<!\w)(?P<number>{_SIGNED_NUMBER})\s*°\s*[CcСс](?!\w)",
    re.IGNORECASE,
)
_UNIT_RE = re.compile(
    rf"(?<!\w)(?P<number>{_SIGNED_NUMBER})\s*"
    r"(?P<unit>мл|кг|см|мм|минут(?:а|ы|у|е|ой)?|секунд(?:а|ы|у|е|ой)?|"
    r"час(?:а|ов)?|л|г|м)(?!\w)",
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
    "мл": (("миллилитр", "миллилитра", "миллилитров"), "masculine"),
    "л": (("литр", "литра", "литров"), "masculine"),
    "г": (("грамм", "грамма", "граммов"), "masculine"),
    "кг": (("килограмм", "килограмма", "килограммов"), "masculine"),
    "м": (("метр", "метра", "метров"), "masculine"),
    "см": (("сантиметр", "сантиметра", "сантиметров"), "masculine"),
    "мм": (("миллиметр", "миллиметра", "миллиметров"), "masculine"),
    "минут": (("минута", "минуты", "минут"), "feminine"),
    "секунд": (("секунда", "секунды", "секунд"), "feminine"),
    "час": (("час", "часа", "часов"), "masculine"),
}


@dataclass
class _Protector:
    values: list[str] = field(default_factory=list)

    def protect(self, text: str, pattern: Pattern[str]) -> str:
        def replace(match: Match[str]) -> str:
            index = len(self.values)
            self.values.append(match.group(0))
            return f"\ue000{chr(0xE100 + index)}\ue001"

        return pattern.sub(replace, text)

    def restore(self, text: str) -> str:
        for index, value in enumerate(self.values):
            text = text.replace(f"\ue000{chr(0xE100 + index)}\ue001", value)
        return text


def _plural_index(number: int) -> int:
    number = abs(number)
    if number % 100 in range(11, 15):
        return 2
    if number % 10 == 1:
        return 0
    if number % 10 in range(2, 5):
        return 1
    return 2


def _under_thousand(number: int, gender: str = "masculine") -> list[str]:
    words: list[str] = []
    hundreds, remainder = divmod(number, 100)
    if hundreds:
        words.append(_HUNDREDS[hundreds * 100])
    if remainder in _TEENS:
        words.append(_TEENS[remainder])
    else:
        tens, ones = divmod(remainder, 10)
        if tens:
            words.append(_TENS[tens * 10])
        if ones:
            words.append(_ONES[gender][ones])
    return words


def russian_cardinal(number: int, gender: str = "masculine") -> str:
    """Return a Russian cardinal with unit-aware one and two forms."""

    if number == 0:
        return _ONES[gender][0]
    if number < 0:
        return f"минус {russian_cardinal(-number, gender)}"

    words: list[str] = []
    remainder = number
    for scale_value, forms, scale_gender in _SCALES:
        group, remainder = divmod(remainder, scale_value)
        if group:
            words.extend(_integer_words(group, scale_gender))
            words.append(forms[_plural_index(group)])
    words.extend(_under_thousand(remainder, gender))
    return " ".join(words)


def _integer_words(number: int, gender: str = "masculine") -> list[str]:
    return russian_cardinal(number, gender).split()


def _under_thousand_genitive(number: int, gender: str = "masculine") -> list[str]:
    words: list[str] = []
    hundreds, remainder = divmod(number, 100)
    if hundreds:
        words.append(_GENITIVE_HUNDREDS[hundreds * 100])
    if remainder in _GENITIVE_TEENS:
        words.append(_GENITIVE_TEENS[remainder])
    else:
        tens, ones = divmod(remainder, 10)
        if tens:
            words.append(_GENITIVE_TENS[tens * 10])
        if ones:
            words.append(_GENITIVE_ONES[gender][ones])
    return words


def russian_genitive(number: int, gender: str = "masculine") -> str:
    """Return the numeral form used after Russian ``от`` and ``до``."""

    if number == 0:
        return _GENITIVE_ONES[gender][0]
    if number < 0:
        return f"минус {russian_genitive(-number, gender)}"

    words: list[str] = []
    remainder = number
    for scale_value, forms, scale_gender in _SCALES:
        group, remainder = divmod(remainder, scale_value)
        if group:
            words.extend(_genitive_words(group, scale_gender))
            words.append(forms[1] if _plural_index(group) == 0 else forms[2])
    words.extend(_under_thousand_genitive(remainder, gender))
    return " ".join(words)


def _genitive_words(number: int, gender: str = "masculine") -> list[str]:
    return russian_genitive(number, gender).split()


def russian_ordinal_genitive(number: int) -> str:
    """Return a masculine genitive ordinal, primarily for spoken dates."""

    if number in _ORDINAL_GENITIVE:
        return _ORDINAL_GENITIVE[number]
    if number < 100:
        tens = number // 10 * 10
        return f"{_TENS[tens]} {_ORDINAL_GENITIVE[number % 10]}"
    if number < 1_000:
        hundreds = number // 100 * 100
        remainder = number % 100
        return f"{_HUNDREDS[hundreds]} {russian_ordinal_genitive(remainder)}"
    if number < 10_000:
        thousands = number // 1_000 * 1_000
        remainder = number % 1_000
        if remainder:
            prefix = russian_cardinal(thousands)
            if prefix == "одна тысяча":
                prefix = "тысяча"
            return f"{prefix} {russian_ordinal_genitive(remainder)}"
    raise ValueError("Russian date ordinals support years from 1 through 9999")


def _parse_integer(value: str) -> int:
    compact = re.sub(r"[ \u00a0\u202f]", "", value)
    return int(compact.replace("−", "-"))


def _number_words(value: str, gender: str = "masculine") -> str:
    normalized = value.replace("−", "-")
    sign = ""
    if normalized.startswith("-"):
        sign = "минус "
        normalized = normalized[1:]

    if "." not in normalized and "," not in normalized:
        return sign + russian_cardinal(_parse_integer(normalized), gender)

    integer_part, fraction_part = re.split(r"[.,]", normalized, maxsplit=1)
    denominator = _DECIMAL_DENOMINATORS.get(len(fraction_part))
    if denominator is None:
        return value
    integer = _parse_integer(integer_part)
    numerator = int(fraction_part)
    whole_form = "целая" if _plural_index(integer) == 0 else "целых"
    fraction_form = denominator[0] if _plural_index(numerator) == 0 else denominator[1]
    return (
        f"{sign}{russian_cardinal(integer, 'feminine')} {whole_form} "
        f"{russian_cardinal(numerator, 'feminine')} {fraction_form}"
    )


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


def transform_unprotected(text: str, transform: Callable[[str], _T]) -> _T | str:
    """Apply a text transform without touching formulas, URLs, files, or IDs."""

    protector = _Protector()
    protected = _mask_nonlinguistic(text, protector)
    transformed = transform(protected)
    if not isinstance(transformed, str):
        return transformed
    return protector.restore(transformed)


def _replace_date(match: Match[str]) -> str:
    day = int(match.group("day"))
    month = int(match.group("month"))
    year = int(match.group("year"))
    try:
        date(year, month, day)
    except ValueError:
        return match.group(0)
    return (
        f"{russian_ordinal_genitive(day)} {_MONTHS_GENITIVE[month]} "
        f"{russian_ordinal_genitive(year)} года"
    )


def _unit_words(number_text: str, forms: tuple[str, str, str], gender: str) -> str:
    normalized = number_text.replace("−", "-")
    if "." in normalized or "," in normalized:
        unit = forms[1]
    else:
        unit = forms[_plural_index(_parse_integer(normalized))]
    return f"{_number_words(number_text, gender)} {unit}"


def _replace_time(match: Match[str]) -> str:
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    if hour > 23 or minute > 59:
        return match.group(0)
    hours = _unit_words(str(hour), ("час", "часа", "часов"), "masculine")
    minutes = _unit_words(str(minute), ("минута", "минуты", "минут"), "feminine")
    return f"{hours} {minutes}"


def _replace_range(match: Match[str]) -> str:
    start = _parse_integer(match.group("start"))
    end = _parse_integer(match.group("end"))
    return f"от {russian_genitive(start)} до {russian_genitive(end)}"


def _replace_unit(match: Match[str]) -> str:
    unit = match.group("unit").lower()
    _, forms, gender = max(
        (
            (prefix, forms, gender)
            for prefix, (forms, gender) in _UNIT_NAMES.items()
            if unit.startswith(prefix)
        ),
        key=lambda item: len(item[0]),
    )
    return _unit_words(match.group("number"), forms, gender)


def _replace_temperature_after_do(match: Match[str]) -> str:
    number = _parse_integer(match.group("number"))
    degree = "градуса Цельсия" if _plural_index(number) == 0 else "градусов Цельсия"
    return f"{match.group('preposition')}{russian_genitive(number)} {degree}"


def normalize_russian_text(text: str) -> str:
    """Expand supported numeric forms while preserving non-linguistic tokens."""

    if not text or not any(char.isdigit() for char in text):
        return text

    protector = _Protector()
    normalized = _mask_nonlinguistic(text, protector)
    normalized = _DATE_RE.sub(_replace_date, normalized)
    normalized = protector.protect(normalized, _DOTTED_IDENTIFIER_RE)
    normalized = _TIME_RE.sub(_replace_time, normalized)
    normalized = protector.protect(normalized, _TIME_LIKE_RE)
    normalized = _RANGE_RE.sub(_replace_range, normalized)
    normalized = _TEMPERATURE_AFTER_DO_RE.sub(
        _replace_temperature_after_do, normalized
    )
    normalized = _TEMPERATURE_RE.sub(
        lambda match: _unit_words(
            match.group("number"),
            ("градус Цельсия", "градуса Цельсия", "градусов Цельсия"),
            "masculine",
        ),
        normalized,
    )
    normalized = _UNIT_RE.sub(_replace_unit, normalized)
    normalized = _PERCENT_RE.sub(
        lambda match: _unit_words(
            match.group("number"),
            ("процент", "процента", "процентов"),
            "masculine",
        ),
        normalized,
    )
    normalized = _DECIMAL_RE.sub(
        lambda match: _number_words(match.group("number")), normalized
    )
    normalized = _INTEGER_RE.sub(
        lambda match: russian_cardinal(_parse_integer(match.group("number"))),
        normalized,
    )
    return protector.restore(normalized)


__all__ = [
    "normalize_russian_text",
    "russian_cardinal",
    "russian_genitive",
    "russian_ordinal_genitive",
    "transform_unprotected",
]
