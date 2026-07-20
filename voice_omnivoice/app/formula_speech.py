"""Kazakh spoken form for chemical formulas and reaction equations."""

from __future__ import annotations

import re

from .text_normalization import kazakh_cardinal


# One-letter symbols keep their chemistry reading, not the English letter
# names used for acronyms (H is "аш", not "эйч").
CHEMISTRY_LETTER_NAMES_KK = {
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

# Physics variables such as U1 or V2 look exactly like formulas, so a token
# with a single one-letter symbol is read with these letter names instead.
PHYSICS_LETTER_NAMES_KK = {
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

ELEMENT_NAMES_KK = {
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

# Formulas built only from one-letter symbols look exactly like acronyms
# (ISBN, CU, CB, SN, HB), so they are read only when listed here.
# ponytail: explicit allowlist, extend it when the corpus shows another one.
_LETTER_ONLY_FORMULAS = {"KOH", "CO", "NO", "HI", "HF"}

ARROW_WORD_KK = "түзіледі"
PLUS_WORD_KK = "плюс"
MINUS_WORD_KK = "минус"
HYDRATE_WORD_KK = "на"
MULTIPLIER_WORD_KK = "рет"

_SUBSCRIPT_DIGITS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_SUPERSCRIPT_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")

_SUPERSCRIPT_CHARGE_RE = re.compile(r"([⁰¹²³⁴⁵⁶⁷⁸⁹]*)([⁺⁻])$")
_ASCII_CHARGE_RE = re.compile(r"(\d*)([+-])$")
_TOKEN_RE = re.compile(r"[A-Z][a-z]?|\d+|[()\[\]·]")

_OPENING = "(["
_CLOSING = ")]"

_ARROW = r"(?:->|=>|→|⟶|↔|⇄|⇌|⇔)"
_REACTION_PART = r"[A-Za-z0-9()\[\]·₀-₉⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+"
_REACTION_RE = re.compile(
    rf"(?<![\w-]){_REACTION_PART}"
    rf"(?:\s*(?:\+|{_ARROW})\s*{_REACTION_PART})+(?![\w-])"
)
_ARROW_SPLIT_RE = re.compile(rf"(\+|{_ARROW})")
_TRIM_RE = re.compile(r"^(?P<lead>[.\s]*)(?P<core>.*?)(?P<tail>[.\s]*)$", re.S)


def _element_name(symbol: str, physics: bool) -> str | None:
    if len(symbol) == 1:
        table = PHYSICS_LETTER_NAMES_KK if physics else CHEMISTRY_LETTER_NAMES_KK
        return table.get(symbol)
    return ELEMENT_NAMES_KK.get(symbol)


def _split_charge(text: str) -> tuple[str, tuple[str, str] | None]:
    match = _SUPERSCRIPT_CHARGE_RE.search(text)
    if match:
        digits = match.group(1).translate(_SUPERSCRIPT_DIGITS)
        sign = "+" if match.group(2) == "⁺" else "-"
        return text[: match.start()], (digits, sign)
    match = _ASCII_CHARGE_RE.search(text)
    if match:
        return text[: match.start()], (match.group(1), match.group(2))
    return text, None


def _tokenize(body: str) -> list[str] | None:
    tokens = _TOKEN_RE.findall(body)
    if "".join(tokens) != body:
        return None
    return tokens


def _is_plausible(
    value: str, tokens: list[str], symbols: list[str], charged: bool
) -> bool:
    # No formula repeats a symbol back to back, but identifiers like FF12 do.
    if any(first == second for first, second in zip(tokens, tokens[1:])):
        return False
    letters_only = (
        not charged
        and tokens == symbols
        and all(len(symbol) == 1 for symbol in symbols)
    )
    return not letters_only or value in _LETTER_ONLY_FORMULAS


def _render(value: str) -> str | None:
    text = value.strip()
    if not text:
        return None
    body, charge = _split_charge(text)
    tokens = _tokenize(body.translate(_SUBSCRIPT_DIGITS))
    if not tokens:
        return None

    symbols = [token for token in tokens if token[0].isalpha()]
    if not symbols:
        return None
    # Element check first: it is what declines identifiers such as 25A whose
    # letters only exist in the physics table.
    if any(_element_name(symbol, physics=False) is None for symbol in symbols):
        return None
    if not _is_plausible(text, tokens, symbols, charge is not None):
        return None
    physics = len(symbols) == 1 and len(symbols[0]) == 1

    words: list[str] = []
    after_group = False
    for token in tokens:
        if token in _OPENING or token in _CLOSING:
            after_group = token in _CLOSING
            continue
        if token == "·":
            words.append(HYDRATE_WORD_KK)
            after_group = False
            continue
        if token.isdigit():
            spoken = kazakh_cardinal(int(token))
            words.append(
                f"{spoken} {MULTIPLIER_WORD_KK}" if after_group else spoken
            )
            after_group = False
            continue
        name = _element_name(token, physics)
        if name is None:
            return None
        words.append(name)
        after_group = False

    if charge is not None:
        digits, sign = charge
        if digits:
            words.append(kazakh_cardinal(int(digits)))
        words.append(PLUS_WORD_KK if sign == "+" else MINUS_WORD_KK)
    return " ".join(words)


def speak_formula(value: str) -> str:
    """Return the Kazakh spoken form of a formula, or the input unchanged."""

    try:
        # The candidate pattern swallows sentence punctuation, so peel it off
        # and put it back around the spoken form.
        trimmed = _TRIM_RE.match(value)
        if trimmed is None:
            return value
        spoken = _render(trimmed.group("core"))
        if not spoken:
            return value
        return f"{trimmed.group('lead')}{spoken}{trimmed.group('tail')}"
    except Exception:  # pragma: no cover - TTS hot path must never raise
        return value


def speak_reaction(value: str) -> str:
    """Return the spoken form of a reaction, or the input unchanged."""

    try:
        pieces = _ARROW_SPLIT_RE.split(value)
        if len(pieces) < 3:
            return value
        words: list[str] = []
        for index, piece in enumerate(pieces):
            if index % 2:
                words.append(PLUS_WORD_KK if piece == "+" else ARROW_WORD_KK)
                continue
            spoken = _render(piece)
            if spoken is None:
                return value
            words.append(spoken)
        return " ".join(words)
    except Exception:  # pragma: no cover - TTS hot path must never raise
        return value


__all__ = [
    "CHEMISTRY_LETTER_NAMES_KK",
    "ELEMENT_NAMES_KK",
    "PHYSICS_LETTER_NAMES_KK",
    "speak_formula",
    "speak_reaction",
]
