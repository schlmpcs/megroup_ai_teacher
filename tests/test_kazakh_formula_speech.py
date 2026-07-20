"""Tests for Kazakh chemical formula and reaction speech."""

import re

import pytest

from voice_omnivoice.app.abbreviation_normalization import (
    normalize_kazakh_tts_text,
)
from voice_omnivoice.app.formula_speech import speak_formula, speak_reaction


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("H2O", "аш екі о"),
        ("H₂O", "аш екі о"),
        ("CO2", "це о екі"),
        ("CO₂", "це о екі"),
        ("NaCl", "натрий хлор"),
        ("NaOH", "натрий о аш"),
        ("KOH", "калий о аш"),
        ("HCl", "аш хлор"),
        ("H2SO4", "аш екі эс о төрт"),
        ("CuSO4", "купрум эс о төрт"),
        ("CaCO3", "кальций це о үш"),
        ("NaHCO3", "натрий аш це о үш"),
        ("KMnO4", "калий марганец о төрт"),
        ("AgNO3", "аргентум эн о үш"),
        ("C6H12O6", "це алты аш он екі о алты"),
        ("Ca(OH)2", "кальций о аш екі рет"),
        ("Al2(SO4)3", "алюминий екі эс о төрт үш рет"),
        ("CuSO4·5H2O", "купрум эс о төрт на бес аш екі о"),
        ("2H2O", "екі аш екі о"),
        ("2H2", "екі аш екі"),
    ],
)
def test_kazakh_formulas_are_spoken_symbol_by_symbol(source, expected):
    assert speak_formula(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("KOH", "калий о аш"),
        ("CO", "це о"),
        ("NO", "эн о"),
        ("HI", "аш йод"),
        ("HF", "аш фтор"),
    ],
)
def test_kazakh_allowlisted_letter_only_formulas(source, expected):
    assert speak_formula(source) == expected


@pytest.mark.parametrize(
    "source",
    [
        "CU",
        "CB",
        "SN",
        "HB",
        "PD",
        "BA",
        "MN",
        "NA",
        "LI",
        "IR",
        "FR",
        "XC",
    ],
)
def test_kazakh_uppercase_abbreviations_are_not_formulas(source):
    assert speak_formula(source) == source


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Ca2+", "кальций екі плюс"),
        ("Ca²⁺", "кальций екі плюс"),
        ("SO4²⁻", "эс о төрт екі минус"),
        ("H+", "аш плюс"),
    ],
)
def test_kazakh_ion_charges(source, expected):
    assert speak_formula(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("U1", "у бір"),
        ("U2", "у екі"),
        ("V1", "вэ бір"),
        ("V2", "вэ екі"),
        ("F1", "эф бір"),
        ("F2", "эф екі"),
        ("S1", "эс бір"),
        ("S2", "эс екі"),
        ("I2", "и екі"),
        ("N0", "эн нөл"),
        ("B0", "бэ нөл"),
        ("K1", "ка бір"),
        ("C1", "цэ бір"),
        ("W2", "дубль-вэ екі"),
    ],
)
def test_kazakh_single_symbol_tokens_stay_physics_variables(source, expected):
    assert speak_formula(source) == expected


@pytest.mark.parametrize(
    "source",
    [
        "OKULYK",
        "KWWSV",
        "FF12",
        "ISBN",
        "pH",
        "DNA",
        "25A",
        "LAB",
        "2",
        "",
    ],
)
def test_kazakh_non_formulas_pass_through_unchanged(source):
    assert speak_formula(source) == source


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "2H₂ + O₂ → 2H₂O",
            "екі аш екі плюс о екі түзіледі екі аш екі о",
        ),
        (
            "2H2 + O2 -> 2H2O",
            "екі аш екі плюс о екі түзіледі екі аш екі о",
        ),
        (
            "NaOH + HCl → NaCl + H2O",
            "натрий о аш плюс аш хлор түзіледі натрий хлор плюс аш екі о",
        ),
    ],
)
def test_kazakh_reactions_are_spoken(source, expected):
    assert speak_reaction(source) == expected


@pytest.mark.parametrize(
    "source",
    [
        "2 + 2",
        "x + y",
        "a -> b",
        "H2O",
    ],
)
def test_kazakh_non_reactions_pass_through_unchanged(source):
    assert speak_reaction(source) == source


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "H2O молекуласында 2 сутек атомы бар.",
            "аш екі о молекуласында екі сутек атомы бар.",
        ),
        (
            "Ca(OH)2 ерітіндісіне CuSO4 қосыңыз.",
            "кальций о аш екі рет ерітіндісіне купрум эс о төрт қосыңыз.",
        ),
        (
            "2H₂ + O₂ → 2H₂O реакциясы",
            "екі аш екі плюс о екі түзіледі екі аш екі о реакциясы",
        ),
        (
            "U1 және U2 кернеуін өлшеңіз.",
            "у бір және у екі кернеуін өлшеңіз.",
        ),
        ("sample_25A", "sample_25A"),
        ("25A", "25A"),
    ],
)
def test_kazakh_pipeline_speaks_formulas(source, expected):
    assert normalize_kazakh_tts_text(source) == expected


def test_kazakh_pipeline_keeps_units_and_formulas_apart():
    # Nested protectors used to mint the same placeholder, so the unit phrase
    # overwrote the formula span.
    spoken = normalize_kazakh_tts_text(
        "Сынауыққа 5 мл CuSO4 ерітіндісін құйыңыз."
    )
    assert spoken == (
        "Сынауыққа бес миллилитр купрум эс о төрт ерітіндісін құйыңыз."
    )
    assert "миллилитр миллилитр" not in spoken


@pytest.mark.parametrize(
    "source",
    [
        "H2O CO2 NaCl KMnO4 CuSO4·5H2O Ca(OH)2",
        "2H₂ + O₂ → 2H₂O",
        "NaOH ерітіндісі және H2SO4 қышқылы",
    ],
)
def test_kazakh_pipeline_leaves_no_bare_latin(source):
    assert not re.search(r"[A-Za-z]", normalize_kazakh_tts_text(source))


@pytest.mark.parametrize(
    "source,expected",
    [
        ("Ca2+ иондары.", "кальций екі плюс иондары."),
        ("Глюкоза C6H12O6.", "Глюкоза це алты аш он екі о алты."),
        ("Бөлінеді CO.", "Бөлінеді це о."),
    ],
)
def test_kazakh_pipeline_speaks_ions_and_sentence_final_formulas(source, expected):
    """An ASCII charge must beat the math pattern, a trailing dot must not hide
    a formula."""

    assert normalize_kazakh_tts_text(source) == expected


@pytest.mark.parametrize("source", ["LAB-204 белгісі.", "report2.csv файлы."])
def test_kazakh_pipeline_ion_pass_ignores_identifiers(source):
    assert normalize_kazakh_tts_text(source) == source
