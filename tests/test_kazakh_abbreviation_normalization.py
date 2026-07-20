"""Tests for Kazakh TTS abbreviations, initials, and letter names."""

import pytest

from voice_omnivoice.app.abbreviation_normalization import (
    KAZAKH_ABBREVIATIONS,
    KAZAKH_ACRONYMS,
    KAZAKH_UNIT_ABBREVIATIONS,
    normalize_kazakh_abbreviations,
    normalize_kazakh_tts_text,
)
from voice_omnivoice.app.text_normalization import normalize_kazakh_text


def _normalize_pipeline(text: str) -> str:
    return normalize_kazakh_tts_text(text)


def test_kazakh_lexicons_contain_required_entries():
    assert {"т.б.", "т.с.с.", "т.с.", "ж.", "б."} <= (
        KAZAKH_ABBREVIATIONS.keys()
    )
    assert {
        "кг",
        "г",
        "мг",
        "мл",
        "см",
        "км",
        "мин",
        "сағ",
        "Н",
        "Дж",
        "Вт",
        "Па",
        "Гц",
        "°C",
    } <= KAZAKH_UNIT_ABBREVIATIONS.keys()
    assert {"AI", "VR", "API", "pH", "DNA"} <= KAZAKH_ACRONYMS.keys()


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("т.б.", "тағы басқа"),
        ("т.с.с.", "тағы сол сияқты"),
        ("т.с.", "тағы сол"),
        ("2026 ж.", "2026 жыл"),
        ("5 б.", "5 бет"),
    ],
)
def test_kazakh_clear_abbreviations(source, expected):
    assert normalize_kazakh_abbreviations(source) == expected


def test_kazakh_standalone_science_unit_abbreviations():
    source = "кг г мг мл см км мин сағ Н Дж Вт Па Гц °C"
    expected = (
        "килограмм грамм миллиграмм миллилитр сантиметр "
        "километр минут сағат ньютон джоуль ватт паскаль "
        "герц градус Цельсий"
    )
    assert normalize_kazakh_abbreviations(source) == expected


def test_kazakh_new_units_feed_the_existing_number_normalizer():
    source = "2 кг, 5 мг, 3 км, 10 мин, 2 сағ, 25 °C, 5 Н, 20 м/с"
    expected = (
        "екі килограмм, бес миллиграмм, "
        "үш километр, он минут, "
        "екі сағат, жиырма бес градус Цельсий, бес ньютон, "
        "секундына жиырма метр"
    )
    assert _normalize_pipeline(source) == expected


@pytest.mark.parametrize(
    "source",
    [
        "7-10",
        "12.03.2026",
        "14:30",
        "3.14",
        "25%",
        "2 кг",
        "x2 2",
        "31.02.2026 25:99 version 1.2.3",
    ],
)
def test_kazakh_pipeline_preserves_existing_number_behavior(source):
    assert _normalize_pipeline(source) == normalize_kazakh_text(source)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("А. Байтұрсынұлы", "а Байтұрсынұлы"),
        ("А. Қ. Байтұрсынұлы", "а қы Байтұрсынұлы"),
        ("А.Қ. Байтұрсынұлы", "а қы Байтұрсынұлы"),
        ("А. Қ. Б. Байтұрсынұлы", "а қы бэ Байтұрсынұлы"),
    ],
)
def test_kazakh_initials_with_and_without_spaces(source, expected):
    assert normalize_kazakh_abbreviations(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Қ", "қы"),
        ("әрпі Қ", "әрпі қы"),
        ("AI VR API pH DNA", "эй ай ви ар эй пи ай пэ аш ди эн эй"),
        ("GPU ҚР", "джи пи ю қы эр"),
        ("CO", "це о"),
    ],
)
def test_kazakh_letters_and_acronyms(source, expected):
    assert normalize_kazakh_abbreviations(source) == expected


@pytest.mark.parametrize(
    "source",
    [
        "x = 2; 2 + 2 = 4; 2+2; x^2; F=ma; айнымалы Қ; айнымалы API",
        "https://example.com/AI/lab2?q=25 www.example.kz/API/2026",
        "report_AI.csv AI.c lesson_DNA_12.03.2026.txt /labs/API/report.csv",
        "LAB-204 physics-10-kk-02 sample_25A ID: API-42 LAB-ALPHA",
    ],
)
def test_kazakh_pipeline_preserves_protected_tokens(source):
    assert _normalize_pipeline(source) == source
