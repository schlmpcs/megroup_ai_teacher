"""Spoken Russian readings for chemical formulas and reaction equations.

Formulas are read symbol by symbol (``H2O`` is «аш два о»), which covers any
formula without a compound-name dictionary and matches what the student sees on
the VR screen. Every entry point returns its input unchanged when the value does
not parse as chemistry and never raises: this runs on the TTS hot path.
"""

from __future__ import annotations

import re

from .text_normalization import russian_cardinal


# Latin-derived element names used when reading formulas aloud in Russian.
ELEMENT_NAMES_RU = {
    "Ac": "актиний",
    "Ag": "аргентум",
    "Al": "алюминий",
    "Am": "америций",
    "Ar": "аргон",
    "As": "арсеникум",
    "At": "астат",
    "Au": "аурум",
    "Ba": "барий",
    "Be": "бериллий",
    "Bh": "борий",
    "Bi": "висмут",
    "Bk": "берклий",
    "Br": "бром",
    "Ca": "кальций",
    "Cd": "кадмий",
    "Ce": "церий",
    "Cf": "калифорний",
    "Cl": "хлор",
    "Cm": "кюрий",
    "Cn": "коперниций",
    "Co": "кобальт",
    "Cr": "хром",
    "Cs": "цезий",
    "Cu": "купрум",
    "Db": "дубний",
    "Ds": "дармштадтий",
    "Dy": "диспрозий",
    "Er": "эрбий",
    "Es": "эйнштейний",
    "Eu": "европий",
    "Fe": "феррум",
    "Fl": "флеровий",
    "Fm": "фермий",
    "Fr": "франций",
    "Ga": "галлий",
    "Gd": "гадолиний",
    "Ge": "германий",
    "He": "гелий",
    "Hf": "гафний",
    "Hg": "гидраргирум",
    "Ho": "гольмий",
    "Hs": "хассий",
    "In": "индий",
    "Ir": "иридий",
    "Kr": "криптон",
    "La": "лантан",
    "Li": "литий",
    "Lr": "лоуренсий",
    "Lu": "лютеций",
    "Lv": "ливерморий",
    "Mc": "московий",
    "Md": "менделевий",
    "Mg": "магний",
    "Mn": "марганец",
    "Mo": "молибден",
    "Mt": "мейтнерий",
    "Na": "натрий",
    "Nb": "ниобий",
    "Nd": "неодим",
    "Ne": "неон",
    "Nh": "нихоний",
    "Ni": "никель",
    "No": "нобелий",
    "Np": "нептуний",
    "Og": "оганесон",
    "Os": "осмий",
    "Pa": "протактиний",
    "Pb": "плюмбум",
    "Pd": "палладий",
    "Pm": "прометий",
    "Po": "полоний",
    "Pr": "празеодим",
    "Pt": "платина",
    "Pu": "плутоний",
    "Ra": "радий",
    "Rb": "рубидий",
    "Re": "рений",
    "Rf": "резерфордий",
    "Rg": "рентгений",
    "Rh": "родий",
    "Rn": "радон",
    "Ru": "рутений",
    "Sb": "стибиум",
    "Sc": "скандий",
    "Se": "селен",
    "Sg": "сиборгий",
    "Si": "силиций",
    "Sm": "самарий",
    "Sn": "станнум",
    "Sr": "стронций",
    "Ta": "тантал",
    "Tb": "тербий",
    "Tc": "технеций",
    "Te": "теллур",
    "Th": "торий",
    "Ti": "титан",
    "Tl": "таллий",
    "Tm": "тулий",
    "Ts": "теннессин",
    "Xe": "ксенон",
    "Yb": "иттербий",
    "Zn": "цинк",
    "Zr": "цирконий",
}

# One-letter symbols as a chemist reads them, not as English acronym letters.
CHEMISTRY_LETTER_NAMES_RU = {
    "B": "бор",
    "C": "це",
    "F": "фтор",
    "H": "аш",
    "I": "йод",
    "K": "калий",
    "N": "эн",
    "O": "о",
    "P": "пэ",
    "S": "эс",
    "U": "уран",
    "V": "ванадий",
    "W": "вольфрам",
    "Y": "иттрий",
}

# Russian names of Latin letters as used for physics variables such as U1, V2.
PHYSICS_LETTER_NAMES_RU = {
    "A": "а",
    "B": "бэ",
    "C": "цэ",
    "D": "дэ",
    "E": "е",
    "F": "эф",
    "G": "гэ",
    "H": "аш",
    "I": "и",
    "J": "жи",
    "K": "ка",
    "L": "эль",
    "M": "эм",
    "N": "эн",
    "O": "о",
    "P": "пэ",
    "Q": "ку",
    "R": "эр",
    "S": "эс",
    "T": "тэ",
    "U": "у",
    "V": "вэ",
    "W": "дубль-вэ",
    "X": "икс",
    "Y": "игрек",
    "Z": "зет",
}

_CHEMISTRY_NAMES = {**ELEMENT_NAMES_RU, **CHEMISTRY_LETTER_NAMES_RU}
_MULTIPLIER_NAMES = {2: "дважды", 3: "трижды", 4: "четырежды"}

# Formulas built only from one-letter symbols look exactly like the corpus
# abbreviations (CU, CB, SN, HB, ISBN, KWWSV), so they are read only when
# listed here.
# ponytail: this allowlist is the deliberate ceiling, a new digitless formula
# gets an entry here rather than a looser heuristic.
_LETTER_ONLY_FORMULAS = {"KOH", "CO", "NO", "HI", "HF"}

_SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_SUPERSCRIPT_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
_SUPERSCRIPT_CHARGE_RE = re.compile(r"[⁰¹²³⁴-⁹]*[⁺⁻]")
_CHARGE_RE = re.compile(r"\^?(\d*)([+\-−])$")
_LEADING_COEFFICIENT_RE = re.compile(r"^\d+")
_TOKEN_RE = re.compile(r"[A-Z][a-z]?|\d+|[()\[\]·]")
_TRIM_RE = re.compile(r"^(?P<lead>[.\s]*)(?P<core>.*?)(?P<tail>[.\s]*)$", re.S)
_SEPARATOR_RE = re.compile(
    r"\s*(<->|->|=>|[+→⟶↔⇄])\s*"
)


def _fold(value: str) -> str:
    """Fold Unicode subscripts to digits and superscript charges to ``^N±``."""

    folded = _SUPERSCRIPT_CHARGE_RE.sub(
        lambda match: "^"
        + match.group(0)[:-1].translate(_SUPERSCRIPT_MAP)
        + ("+" if match.group(0)[-1] == "⁺" else "-"),
        value,
    )
    return folded.translate(_SUBSCRIPT_MAP)


def _parse(value: str) -> tuple[list[str], str, tuple[str, str] | None] | None:
    text = _fold(value.strip())
    if not text:
        return None

    charge: tuple[str, str] | None = None
    match = _CHARGE_RE.search(text)
    if match:
        charge = (match.group(1), match.group(2))
        text = text[: match.start()]

    coefficient = ""
    lead = _LEADING_COEFFICIENT_RE.match(text)
    if lead:
        coefficient = lead.group(0)
        text = text[lead.end() :]

    tokens = _TOKEN_RE.findall(text)
    if not tokens or "".join(tokens) != text:
        return None
    return tokens, coefficient, charge


def _multiplier(number: int) -> str:
    return _MULTIPLIER_NAMES.get(number) or russian_cardinal(number)


def _analyze(value: str) -> tuple[str, bool] | None:
    """Return the spoken form plus whether it was read as a real formula."""

    parsed = _parse(value)
    if parsed is None:
        return None
    tokens, coefficient, charge = parsed

    symbols = [token for token in tokens if token[0].isalpha()]
    if not symbols or any(symbol not in _CHEMISTRY_NAMES for symbol in symbols):
        return None
    # No formula repeats a symbol back to back, but identifiers like FF12 do.
    if any(first == second for first, second in zip(tokens, tokens[1:])):
        return None

    letters_only = not charge and not coefficient and symbols == tokens
    if (
        letters_only
        and not any(len(symbol) == 2 for symbol in symbols)
        and value not in _LETTER_ONLY_FORMULAS
    ):
        return None

    # One symbol means a physics variable with an index (U1, V2, F2); two or
    # more means a real formula read with chemistry names (KOH, H2O).
    is_formula = len(symbols) > 1 or len(symbols[0]) == 2
    names = _CHEMISTRY_NAMES if is_formula else PHYSICS_LETTER_NAMES_RU

    words: list[str] = []
    if coefficient:
        words.append(russian_cardinal(int(coefficient)))
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if token in "([":
            continue
        if token in ")]":
            if index < len(tokens) and tokens[index].isdigit():
                words.append(_multiplier(int(tokens[index])))
                index += 1
            continue
        if token == "·":
            words.append("на")
        elif token.isdigit():
            words.append(russian_cardinal(int(token)))
        else:
            words.append(names[token])
    if charge:
        magnitude, sign = charge
        if magnitude not in ("", "1"):
            words.append(russian_cardinal(int(magnitude)))
        words.append("плюс" if sign == "+" else "минус")
    return " ".join(words), is_formula


def speak_formula(value: str) -> str:
    """Read a single formula token aloud, or return it unchanged."""

    try:
        trimmed = _TRIM_RE.match(value)
        if trimmed is None:
            return value
        spoken = _analyze(trimmed.group("core"))
        if spoken is None:
            return value
        return f"{trimmed.group('lead')}{spoken[0]}{trimmed.group('tail')}"
    except Exception:  # pragma: no cover - defensive, TTS must never fail here
        return value


def speak_reaction(value: str) -> str:
    """Read formulas joined by ``+`` and arrows aloud, or return them unchanged."""

    try:
        pieces = _SEPARATOR_RE.split(value.strip())
        if len(pieces) < 3:
            return value
        words: list[str] = []
        has_formula = False
        for index, piece in enumerate(pieces):
            if index % 2:
                words.append("плюс" if piece == "+" else "образуется")
                continue
            spoken = _analyze(piece)
            if spoken is None:
                return value
            words.append(spoken[0])
            has_formula = has_formula or spoken[1]
        # A sum of physics variables is not a reaction, leave it to math handling.
        if not has_formula:
            return value
        return " ".join(words)
    except Exception:  # pragma: no cover - defensive, TTS must never fail here
        return value


__all__ = [
    "CHEMISTRY_LETTER_NAMES_RU",
    "ELEMENT_NAMES_RU",
    "PHYSICS_LETTER_NAMES_RU",
    "speak_formula",
    "speak_reaction",
]
