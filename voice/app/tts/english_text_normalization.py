"""Deterministic English classroom text normalization for local TTS."""

from __future__ import annotations

import re
from datetime import date
from typing import Match

from .text_normalization import (
    _DOTTED_IDENTIFIER_RE,
    _Protector,
    _mask_nonlinguistic,
)

_ONES = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS = (
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
)
_SCALES = (
    (1_000_000_000_000, "trillion"),
    (1_000_000_000, "billion"),
    (1_000_000, "million"),
    (1_000, "thousand"),
)
_ORDINALS = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
    10: "tenth",
    11: "eleventh",
    12: "twelfth",
    13: "thirteenth",
    14: "fourteenth",
    15: "fifteenth",
    16: "sixteenth",
    17: "seventeenth",
    18: "eighteenth",
    19: "nineteenth",
    20: "twentieth",
    21: "twenty first",
    22: "twenty second",
    23: "twenty third",
    24: "twenty fourth",
    25: "twenty fifth",
    26: "twenty sixth",
    27: "twenty seventh",
    28: "twenty eighth",
    29: "twenty ninth",
    30: "thirtieth",
    31: "thirty first",
}
_MONTHS = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}

_SIGNED_INTEGER = r"[−-]?\d+"
_SIGNED_NUMBER = rf"{_SIGNED_INTEGER}(?:[.,]\d+)?"
_ISO_DATE_RE = re.compile(
    r"(?<![\w-])(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})(?![\w-])"
)
_DOTTED_DATE_RE = re.compile(
    r"(?<![\w.])(?P<day>\d{1,2})[./](?P<month>\d{1,2})[./](?P<year>\d{4})(?!\w|\.\d)"
)
_TIME_RE = re.compile(
    r"(?<![\d:])(?P<hour>\d{1,2}):(?P<minute>\d{2})(?![\d:])"
)
_TEMPERATURE_RE = re.compile(
    rf"(?<!\w)(?P<number>{_SIGNED_NUMBER})\s*°\s*(?P<scale>[CF])(?!\w)",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(rf"(?<!\w)(?P<number>{_SIGNED_NUMBER})\s*%(?!\w)")
_UNIT_RE = re.compile(
    rf"(?<!\w)(?P<number>{_SIGNED_NUMBER})\s*"
    r"(?P<unit>mL|ml|kg|cm|mm|km|Hz|kHz|MHz|Pa|kPa|mol|sec|mins?|hours?|"
    r"min|ms|s|L|l|g|m|h|V|A|W|J|N)(?![A-Za-z])"
)
_DECIMAL_RE = re.compile(
    rf"(?<![\w.,])(?P<number>{_SIGNED_INTEGER}[.,]\d+)(?!\w|[.,]\d)"
)
_INTEGER_RE = re.compile(rf"(?<![\w.,])(?P<number>{_SIGNED_INTEGER})(?!\w|[.,]\d)")

_UNIT_NAMES = {
    "ml": "milliliter",
    "l": "liter",
    "g": "gram",
    "kg": "kilogram",
    "m": "meter",
    "cm": "centimeter",
    "mm": "millimeter",
    "km": "kilometer",
    "ms": "millisecond",
    "s": "second",
    "sec": "second",
    "min": "minute",
    "mins": "minute",
    "h": "hour",
    "hour": "hour",
    "hours": "hour",
    "hz": "hertz",
    "khz": "kilohertz",
    "mhz": "megahertz",
    "v": "volt",
    "a": "ampere",
    "w": "watt",
    "j": "joule",
    "n": "newton",
    "pa": "pascal",
    "kpa": "kilopascal",
    "mol": "mole",
}


def english_cardinal(number: int) -> str:
    """Return a plain English cardinal suitable for classroom speech."""
    if number < 0:
        return f"minus {english_cardinal(-number)}"
    if number < 20:
        return _ONES[number]
    if number < 100:
        tens, ones = divmod(number, 10)
        return _TENS[tens] + (f" {_ONES[ones]}" if ones else "")
    if number < 1_000:
        hundreds, remainder = divmod(number, 100)
        result = f"{_ONES[hundreds]} hundred"
        return result + (f" {english_cardinal(remainder)}" if remainder else "")
    for scale, name in _SCALES:
        if number >= scale:
            group, remainder = divmod(number, scale)
            result = f"{english_cardinal(group)} {name}"
            return result + (f" {english_cardinal(remainder)}" if remainder else "")
    return str(number)


def _parse_integer(value: str) -> int:
    return int(value.replace("−", "-"))


def _number_words(value: str) -> str:
    normalized = value.replace("−", "-")
    sign = ""
    if normalized.startswith("-"):
        sign = "minus "
        normalized = normalized[1:]
    if "." not in normalized and "," not in normalized:
        return sign + english_cardinal(int(normalized))
    whole, fraction = re.split(r"[.,]", normalized, maxsplit=1)
    spoken_fraction = " ".join(_ONES[int(digit)] for digit in fraction)
    return f"{sign}{english_cardinal(int(whole))} point {spoken_fraction}"


def _date_words(year: int, month: int, day: int, original: str) -> str:
    try:
        date(year, month, day)
    except ValueError:
        return original
    return f"{_MONTHS[month]} {_ORDINALS[day]} {english_cardinal(year)}"


def _replace_date(match: Match[str]) -> str:
    return _date_words(
        int(match.group("year")),
        int(match.group("month")),
        int(match.group("day")),
        match.group(0),
    )


def _replace_time(match: Match[str]) -> str:
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    if hour > 23 or minute > 59:
        return match.group(0)
    if minute == 0:
        return f"{english_cardinal(hour)} o'clock"
    minute_words = (
        f"oh {english_cardinal(minute)}" if minute < 10 else english_cardinal(minute)
    )
    return f"{english_cardinal(hour)} {minute_words}"


def _pluralized(number_text: str, noun: str) -> str:
    normalized = number_text.replace("−", "-").replace(",", ".")
    singular = float(normalized) == 1.0
    suffix = noun if singular or noun in {"hertz", "kilohertz", "megahertz"} else noun + "s"
    return f"{_number_words(number_text)} {suffix}"


def _replace_temperature(match: Match[str]) -> str:
    scale = "Celsius" if match.group("scale").casefold() == "c" else "Fahrenheit"
    number = match.group("number")
    normalized = number.replace("−", "-").replace(",", ".")
    degree = "degree" if float(normalized) == 1.0 else "degrees"
    return f"{_number_words(number)} {degree} {scale}"


def _replace_unit(match: Match[str]) -> str:
    unit = _UNIT_NAMES[match.group("unit").casefold()]
    return _pluralized(match.group("number"), unit)


def normalize_english_text(text: str) -> str:
    """Expand common spoken forms while preserving formulas, URLs, files, and IDs."""
    if not text or not any(char.isdigit() for char in text):
        return text

    protector = _Protector()
    normalized = _ISO_DATE_RE.sub(_replace_date, text)
    normalized = _DOTTED_DATE_RE.sub(_replace_date, normalized)
    normalized = _mask_nonlinguistic(normalized, protector)
    normalized = protector.protect(normalized, _DOTTED_IDENTIFIER_RE)
    normalized = _TIME_RE.sub(_replace_time, normalized)
    normalized = _TEMPERATURE_RE.sub(_replace_temperature, normalized)
    normalized = _PERCENT_RE.sub(
        lambda match: f"{_number_words(match.group('number'))} percent", normalized
    )
    normalized = _UNIT_RE.sub(_replace_unit, normalized)
    normalized = _DECIMAL_RE.sub(
        lambda match: _number_words(match.group("number")), normalized
    )
    normalized = _INTEGER_RE.sub(
        lambda match: english_cardinal(_parse_integer(match.group("number"))),
        normalized,
    )
    return protector.restore(normalized)


__all__ = ["english_cardinal", "normalize_english_text"]
