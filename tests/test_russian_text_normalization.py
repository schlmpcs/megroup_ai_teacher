"""Focused tests for deterministic Russian TTS text normalization."""

import pytest

from voice.app.tts.text_normalization import normalize_russian_text


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("0", "ноль"),
        ("7", "семь"),
        ("25", "двадцать пять"),
        ("100", "сто"),
        ("1 250", "одна тысяча двести пятьдесят"),
        ("-5", "минус пять"),
        ("3.14", "три целых четырнадцать сотых"),
        ("3,14", "три целых четырнадцать сотых"),
        ("25%", "двадцать пять процентов"),
        ("25 °C", "двадцать пять градусов Цельсия"),
        ("5 мл", "пять миллилитров"),
        ("10 г", "десять граммов"),
        ("2 кг", "два килограмма"),
        ("3 м", "три метра"),
        ("15 минут", "пятнадцать минут"),
        (
            "12.03.2026",
            "двенадцатого марта две тысячи двадцать шестого года",
        ),
        ("14:30", "четырнадцать часов тридцать минут"),
        ("7–10", "от семи до десяти"),
        ("7-10", "от семи до десяти"),
    ],
)
def test_supported_russian_numeric_forms(source, expected):
    assert normalize_russian_text(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "Нагрейте воду до 25 °C.",
            "Нагрейте воду до двадцати пяти градусов Цельсия.",
        ),
        (
            "Добавьте 3,14 г соли.",
            "Добавьте три целых четырнадцать сотых грамма соли.",
        ),
        (
            "Опыт начнётся 12.03.2026 в 14:30.",
            "Опыт начнётся двенадцатого марта две тысячи двадцать шестого "
            "года в четырнадцать часов тридцать минут.",
        ),
        (
            "В молекуле H2O есть 2 атома водорода.",
            "В молекуле H2O есть два атома водорода.",
        ),
    ],
)
def test_russian_school_examples(source, expected):
    assert normalize_russian_text(source) == expected


@pytest.mark.parametrize(
    "source",
    [
        "H2O CO2 NaCl Fe2(SO4)3 Ca(OH)2 CuSO4·5H2O x2",
        "LAB-204 physics-10-ru-02 sample_25A",
        "https://example.com/lab2?q=25 www.example.ru/2026",
        "report2.csv lesson_12.03.2026.txt",
        "x = 2; 2 + 2 = 4; F=ma",
        "1e-3 6.02×10^23",
    ],
)
def test_russian_preserves_non_linguistic_tokens(source):
    assert normalize_russian_text(source) == source


def test_russian_preserves_invalid_date_and_time_like_identifiers():
    source = "31.02.2026 25:99 version 1.2.3"
    assert normalize_russian_text(source) == source
