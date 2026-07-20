"""Tests for spoken Russian readings of chemical formulas and reactions."""

import re

import pytest

from voice.app.tts.abbreviation_normalization import normalize_russian_tts_text
from voice.app.tts.formula_speech import (
    CHEMISTRY_LETTER_NAMES_RU,
    ELEMENT_NAMES_RU,
    PHYSICS_LETTER_NAMES_RU,
    speak_formula,
    speak_reaction,
)


def test_letter_tables_cover_chemistry_and_physics_readings():
    assert CHEMISTRY_LETTER_NAMES_RU["H"] == "аш"
    assert CHEMISTRY_LETTER_NAMES_RU["C"] == "це"
    assert CHEMISTRY_LETTER_NAMES_RU["K"] == "калий"
    assert PHYSICS_LETTER_NAMES_RU["U"] == "у"
    assert PHYSICS_LETTER_NAMES_RU["V"] == "вэ"
    assert set(PHYSICS_LETTER_NAMES_RU) == set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    assert {
        "Na": "натрий",
        "Mg": "магний",
        "Al": "алюминий",
        "Si": "силиций",
        "Cl": "хлор",
        "Ca": "кальций",
        "Fe": "феррум",
        "Cu": "купрум",
        "Zn": "цинк",
        "Ag": "аргентум",
        "Pb": "плюмбум",
        "Hg": "гидраргирум",
        "Mn": "марганец",
        "Sn": "станнум",
        "Au": "аурум",
    }.items() <= ELEMENT_NAMES_RU.items()
    assert all(len(symbol) == 2 for symbol in ELEMENT_NAMES_RU)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("H2O", "аш два о"),
        ("CO2", "це о два"),
        ("CO₂", "це о два"),
        ("NaCl", "натрий хлор"),
        ("NaOH", "натрий о аш"),
        ("KOH", "калий о аш"),
        ("HCl", "аш хлор"),
        ("MgO", "магний о"),
        ("H2SO4", "аш два эс о четыре"),
        ("CuSO4", "купрум эс о четыре"),
        ("CaCO3", "кальций це о три"),
        ("NaHCO3", "натрий аш це о три"),
        ("KMnO4", "калий марганец о четыре"),
        ("AgNO3", "аргентум эн о три"),
        ("C6H12O6", "це шесть аш двенадцать о шесть"),
        ("Ca(OH)2", "кальций о аш дважды"),
        ("Al2(SO4)3", "алюминий два эс о четыре трижды"),
        ("Fe2(SO4)3", "феррум два эс о четыре трижды"),
        ("CuSO4·5H2O", "купрум эс о четыре на пять аш два о"),
    ],
)
def test_formulas_are_read_symbol_by_symbol(source, expected):
    assert speak_formula(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Ca2+", "кальций два плюс"),
        ("Ca²⁺", "кальций два плюс"),
        ("SO4²⁻", "эс о четыре два минус"),
    ],
)
def test_ion_charges_are_spoken(source, expected):
    assert speak_formula(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("U1", "у один"),
        ("U2", "у два"),
        ("V1", "вэ один"),
        ("V2", "вэ два"),
        ("F1", "эф один"),
        ("F2", "эф два"),
        ("S1", "эс один"),
        ("S2", "эс два"),
        ("I2", "и два"),
        ("N0", "эн ноль"),
        ("B0", "бэ ноль"),
        ("K1", "ка один"),
        ("C1", "цэ один"),
        ("W2", "дубль-вэ два"),
    ],
)
def test_single_symbol_tokens_stay_physics_variables(source, expected):
    assert speak_formula(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("H2", "аш два"),
        ("O2", "о два"),
        ("N2", "эн два"),
    ],
)
def test_diatomic_molecules_read_the_same_in_both_branches(source, expected):
    assert speak_formula(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("CO", "це о"),
        ("NO", "эн о"),
        ("HI", "аш йод"),
        ("HF", "аш фтор"),
        ("KOH", "калий о аш"),
    ],
)
def test_allowlisted_digitless_formulas_are_spoken(source, expected):
    assert speak_formula(source) == expected


@pytest.mark.parametrize(
    "source",
    ["OKULYK", "KWWSV", "FF12", "ISBN", "pH", "DNA", "LAB-204", "x2", ""],
)
def test_non_chemistry_tokens_pass_through(source):
    assert speak_formula(source) == source
    assert speak_reaction(source) == source


@pytest.mark.parametrize(
    "source",
    ["CU", "CB", "SN", "HB", "PD", "BA", "MN", "NA", "LI", "IR", "FR", "XC"],
)
def test_corpus_abbreviations_are_not_read_as_formulas(source):
    assert speak_formula(source) == source
    assert speak_reaction(source) == source


def test_sentence_final_period_is_kept_outside_the_reading():
    assert speak_formula("H2O.") == "аш два о."
    assert normalize_russian_tts_text("Выделяется CO.") == "Выделяется це о."


def test_nested_protectors_do_not_overwrite_each_other():
    # The unit phrase and the formula are protected by different protectors,
    # which used to mint the same placeholder and clobber the formula.
    spoken = normalize_russian_tts_text("Возьмите 5 мл раствора CuSO4.")
    assert "купрум эс о четыре" in spoken
    assert "миллилитров миллилитров" not in spoken
    assert spoken == "Возьмите пять миллилитров раствора купрум эс о четыре."


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "2H₂ + O₂ → 2H₂O",
            "два аш два плюс о два образуется два аш два о",
        ),
        (
            "2H2 + O2 -> 2H2O",
            "два аш два плюс о два образуется два аш два о",
        ),
        (
            "Fe + CuSO4 → FeSO4 + Cu",
            "феррум плюс купрум эс о четыре образуется "
            "феррум эс о четыре плюс купрум",
        ),
    ],
)
def test_reactions_are_spoken(source, expected):
    assert speak_reaction(source) == expected


@pytest.mark.parametrize("source", ["U1 + U2", "F1 + F2", "ABC + DEF", "H2O"])
def test_reactions_decline_without_a_real_formula(source):
    assert speak_reaction(source) == source


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Формула воды H2O.", "Формула воды аш два о."),
        (
            "Реакция 2H₂ + O₂ → 2H₂O идёт быстро.",
            "Реакция два аш два плюс о два образуется два аш два о идёт быстро.",
        ),
        (
            "Возьмите 5 мл раствора CuSO4.",
            "Возьмите пять миллилитров раствора купрум эс о четыре.",
        ),
        ("Напряжение U1 и U2.", "Напряжение у один и у два."),
    ],
)
def test_pipeline_speaks_formulas(source, expected):
    assert normalize_russian_tts_text(source) == expected


@pytest.mark.parametrize(
    "source",
    [
        "В растворе Ca(OH)2 и NaHCO3.",
        "Смесь CuSO4·5H2O с KMnO4 и AgNO3.",
        "Ион Ca²⁺ и ион SO4²⁻.",
        "2H₂ + O₂ → 2H₂O",
    ],
)
def test_pipeline_leaves_no_bare_latin(source):
    assert not re.search(r"[A-Za-z]", normalize_russian_tts_text(source))


@pytest.mark.parametrize(
    "source,expected",
    [
        ("Ионы Ca2+ в растворе.", "Ионы кальций два плюс в растворе."),
        ("Добавьте Na+ и Cl- в раствор.", "Добавьте натрий плюс и хлор минус в раствор."),
        ("Глюкоза C6H12O6.", "Глюкоза це шесть аш двенадцать о шесть."),
        ("Выделяется CO.", "Выделяется це о."),
    ],
)
def test_pipeline_speaks_ions_and_sentence_final_formulas(source, expected):
    """An ASCII charge must beat the math pattern, a trailing dot must not hide
    a formula."""

    assert normalize_russian_tts_text(source) == expected


@pytest.mark.parametrize("source", ["Метка LAB-204.", "Файл report2.csv."])
def test_pipeline_ion_pass_ignores_identifiers(source):
    assert normalize_russian_tts_text(source) == source
