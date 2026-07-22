"""Tests for Russian TTS abbreviations, initials, and letter names."""

import pytest

from voice.app.tts.abbreviation_normalization import (
    RUSSIAN_ABBREVIATIONS,
    RUSSIAN_ACRONYMS,
    RUSSIAN_UNIT_ABBREVIATIONS,
    normalize_russian_abbreviations,
    normalize_russian_tts_text,
)
from voice.app.tts.text_normalization import normalize_russian_text


def _normalize_pipeline(text: str) -> str:
    return normalize_russian_tts_text(text)


def test_russian_lexicons_contain_required_entries():
    assert {
        "т.е.",
        "т.к.",
        "и т.д.",
        "и т.п.",
        "г.",
        "стр.",
        "рис.",
        "табл.",
    } <= RUSSIAN_ABBREVIATIONS.keys()
    assert {"кг", "г", "мг", "мл", "см", "км", "мин", "ч", "°C"} <= (
        RUSSIAN_UNIT_ABBREVIATIONS.keys()
    )
    assert {"AI", "VR", "API", "pH", "DNA"} <= RUSSIAN_ACRONYMS.keys()


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("т.е.", "то есть"),
        ("т.к.", "так как"),
        ("и т.д.", "и так далее"),
        ("и т.п.", "и тому подобное"),
        ("стр. 5", "страница 5"),
        ("рис. 2", "рисунок 2"),
        ("табл. 3", "таблица 3"),
        ("2026 г.", "2026 год"),
        ("г. Москва", "город Москва"),
    ],
)
def test_russian_clear_abbreviations(source, expected):
    assert normalize_russian_abbreviations(source) == expected


def test_russian_standalone_unit_abbreviations():
    source = "кг г мг мл см км мин ч °C"
    expected = (
        "килограмм грамм миллиграмм миллилитр сантиметр "
        "километр минута час градус Цельсия"
    )
    assert normalize_russian_abbreviations(source) == expected


def test_russian_new_units_feed_the_existing_number_normalizer():
    source = "2 кг, 5 мг, 3 км, 10 мин, 2 ч, 25 °C"
    expected = (
        "два килограмма, пять миллиграммов, "
        "три километра, десять минут, два часа, "
        "двадцать пять градусов Цельсия"
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
        "31.02.2026 25:99 version 1.2.3",
    ],
)
def test_russian_pipeline_preserves_existing_number_behavior(source):
    assert _normalize_pipeline(source) == normalize_russian_text(source)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("А. Пушкин", "а Пушкин"),
        ("А. С. Пушкин", "а эс Пушкин"),
        ("А.С. Пушкин", "а эс Пушкин"),
        ("А. С. Б. Иванов", "а эс бэ Иванов"),
    ],
)
def test_russian_initials_with_and_without_spaces(source, expected):
    assert normalize_russian_abbreviations(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Б", "бэ"),
        ("буква Б", "буква бэ"),
        ("AI VR API pH DNA", "эй ай ви ар эй пи ай пэ аш ди эн эй"),
        ("GPU РФ", "джи пи ю эр эф"),
        ("CO", "це о"),
    ],
)
def test_russian_letters_and_acronyms(source, expected):
    assert normalize_russian_abbreviations(source) == expected


@pytest.mark.parametrize(
    "source",
    [
        "CU OKULYK KWWSV",
        "x = 2; 2 + 2 = 4; 2+2; x^2; F=ma; переменная А; переменная API",
        "https://example.com/AI/lab2?q=25 www.example.ru/API/2026",
        "report_AI.csv AI.c lesson_DNA_12.03.2026.txt /labs/API/report.csv",
        "LAB-204 physics-10-ru-02 sample_25A ID: API-42 LAB-ALPHA",
    ],
)
def test_russian_pipeline_preserves_protected_tokens(source):
    assert _normalize_pipeline(source) == source


@pytest.mark.parametrize(
    "source",
    [
        "В молекуле два атома водорода.",
        "С другой стороны, К сожалению, У нас есть раствор.",
        "И тогда О станет ясно.",
        "Я думаю, что это верно.",
    ],
)
def test_single_letter_prepositions_are_not_spelled_out(source):
    """В, С, К, О, У, А and И are words, not letters to spell."""

    assert normalize_russian_tts_text(source) == source


def test_letter_labels_are_still_spelled_out():
    assert normalize_russian_tts_text("Выберите вариант А.") == "Выберите вариант а."


def test_russian_pipeline_speaks_inline_latex_newton_formula():
    source = (
        r"Формула: \( \mathbf{a} = \frac{\mathbf{F}}{m} \), "
        r"где \( \mathbf{a} \), \( \mathbf{F} \) и \( m \)."
    )
    expected = "Формула: а равно эф делённое на эм, где а, эф и эм."

    assert normalize_russian_tts_text(source) == expected
