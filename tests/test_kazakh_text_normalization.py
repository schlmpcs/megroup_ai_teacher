"""Focused tests for deterministic Kazakh OmniVoice text normalization."""

import pytest

from voice_omnivoice.app.text_normalization import normalize_kazakh_text


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("0", "нөл"),
        ("7", "жеті"),
        ("25", "жиырма бес"),
        ("100", "жүз"),
        ("1 250", "бір мың екі жүз елу"),
        ("-5", "минус бес"),
        ("3.14", "үш бүтін жүзден он төрт"),
        ("3,14", "үш бүтін жүзден он төрт"),
        ("25%", "жиырма бес пайыз"),
        ("25 °C", "жиырма бес градус Цельсий"),
        ("5 мл", "бес миллилитр"),
        ("10 г", "он грамм"),
        ("2 кг", "екі килограмм"),
        ("3 м", "үш метр"),
        ("15 минут", "он бес минут"),
        ("12.03.2026", "екі мың жиырма алтыншы жылғы он екінші наурыз"),
        ("14:30", "он төрт сағат отыз минут"),
        ("7–10", "жетіден онға дейін"),
        ("7-10", "жетіден онға дейін"),
    ],
)
def test_supported_kazakh_numeric_forms(source, expected):
    assert normalize_kazakh_text(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "Суды 25 °C-қа дейін қыздырыңыз.",
            "Суды жиырма бес градус Цельсийге дейін қыздырыңыз.",
        ),
        (
            "Ерітіндіге 3,14 г тұз қосыңыз.",
            "Ерітіндіге үш бүтін жүзден он төрт грамм тұз қосыңыз.",
        ),
        (
            "Тәжірибе 12.03.2026 күні сағат 14:30-да басталады.",
            "Тәжірибе екі мың жиырма алтыншы жылғы он екінші наурыз күні "
            "сағат он төрт отызда басталады.",
        ),
        (
            "H2O молекуласында 2 сутек атомы бар.",
            "H2O молекуласында екі сутек атомы бар.",
        ),
    ],
)
def test_kazakh_school_examples(source, expected):
    assert normalize_kazakh_text(source) == expected


@pytest.mark.parametrize(
    "source",
    [
        "H2O CO2 NaCl Fe2(SO4)3 Ca(OH)2 CuSO4·5H2O x2",
        "LAB-204 physics-10-kk-02 sample_25A",
        "https://example.com/lab2?q=25 www.example.kz/2026",
        "report2.csv lesson_12.03.2026.txt",
        "x = 2; 2 + 2 = 4; F=ma",
        "1e-3 6.02×10^23",
    ],
)
def test_kazakh_preserves_non_linguistic_tokens(source):
    assert normalize_kazakh_text(source) == source


def test_kazakh_preserves_invalid_date_and_time_like_identifiers():
    source = "31.02.2026 25:99 version 1.2.3"
    assert normalize_kazakh_text(source) == source
