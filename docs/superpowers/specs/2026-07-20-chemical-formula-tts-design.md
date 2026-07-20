# Chemical formula speech for Russian and Kazakh TTS

Date: 2026-07-20

## Problem

Chemical formulas reach the TTS models verbatim and are spoken wrong or not at
all.

`_protect_chemical_formulas` in both sidecars detects formulas correctly, stores
the matched text in a `_Protector`, and restores it **verbatim** at the end of
normalization. So the string handed to a TTS backend still literally contains
`H2O`, `NaOH`, `CuSO4`.

What each backend then does with it:

| Path | Behaviour today |
| --- | --- |
| RU / Supertonic (default), RU / MMS | `_transliterate_latin` char-maps it: `H2O` becomes `х2о`, with a bare digit the number normalizer never expanded because the span was protected |
| RU / Qwen | raw Latin passed to the model |
| KK / OmniVoice (default) | raw Latin, no transliteration at all; Kazakh vocabulary has no Latin, so the token is dropped or garbled |
| KK / MMS (fallback) | no normalization of any kind runs on this path |

Corpus evidence (257 files under `Лабораторные физхимбио`): `H2`/`H2O`, `N2`,
`O2`, `NaCl`, `NaOH`, `HCl`, `CuSO4`, `H2SO4`, `CaCO3`, `CaSO4`, `MgO`, `CO2`,
`CO₂`, `NaHCO3`, `AgNO3`, `C6H12O6`, `KMnO4`, `C6H8O7`. Both ASCII digits and
Unicode subscripts occur. 230 reaction arrows (`→`) are present. Ion charges
appear as `Ca 2+`, `Zn 2+`, `Cu 2+`. Nearly every formula occurrence sits inside
**Kazakh** lab text, which is the path with no Latin handling at all.

## Decisions

Decided with the user before implementation:

1. **Reading style: symbol-by-symbol.** No compound-name dictionary. `H₂O` is
   «аш два о», not «вода». Covers any formula including ones nobody listed, and
   maps one-to-one to the formula the student sees on the VR screen.
2. **Scope: formulas + ion charges + reaction arrows and `+` separators.**

## Design

### Hook point

The entire integration is one line in each sidecar's
`_protect_chemical_formulas`:

```python
# before
return protector.protect_value(value)
# after
return protector.protect_value(speak_formula(value))
```

Protection is retained. It is what stops the downstream abbreviation and number
normalizers from re-processing the Cyrillic words we just produced.

### New module `formula_speech.py`

One copy per sidecar:

- `voice/app/tts/formula_speech.py` — Russian
- `voice_omnivoice/app/formula_speech.py` — Kazakh

Duplicated deliberately: the two services are isolated Docker build contexts
(OmniVoice requires Transformers 5.x), exactly like the existing
`text_normalization.py` and `abbreviation_normalization.py` pairs.

Public surface:

```python
def speak_formula(value: str) -> str: ...      # single formula token
def speak_reaction(value: str) -> str: ...     # formulas joined by + and arrows
```

Both return the input **unchanged** when it does not parse as chemistry. Never
raise.

### Parsing

1. Fold Unicode subscripts `₀-₉` to ASCII digits. Fold Unicode superscripts
   `⁰-⁹⁺⁻` into a charge suffix.
2. Tokenize into: element symbol `[A-Z][a-z]?`, digit run, `(`, `)`, `·`,
   charge suffix.
3. If any `[A-Z][a-z]?` token is not a real element symbol, return the input
   unchanged. This is what keeps `OKULYK`, `KWWSV`, `FF12`, `Pag` out.

### Rendering

| Piece | Rule |
| --- | --- |
| two-letter symbol | element name: `Na` «натрий», `Cu` «купрум», `Fe` «феррум», `Mn` «марганец», `Cl` «хлор» |
| one-letter symbol | chemistry letter name: `H` «аш», `C` «це», `O` «о», `S` «эс», `N` «эн», `P` «пэ», `K` «калий», `I` «йод» |
| subscript, coefficient | reuse `russian_cardinal(n)` / `kazakh_cardinal(n)` from the sidecar's `text_normalization` — do not reimplement number words |
| `(...)` multiplier | RU 2 «дважды», 3 «трижды», 4 «четырежды», else cardinal. KK «<cardinal> рет» |
| charge `²⁺`, `2+`, `+` | RU «два плюс» / «плюс». KK «екі плюс» / «плюс» |
| `·` in hydrates | RU «на», KK «на» |
| `→ ⟶ ↔ ⇄ ->` | RU «образуется», KK «түзіледі» |
| `+` between formulas | «плюс» |

Do **not** reuse `LATIN_LETTER_NAMES_RU` / `LATIN_LETTER_NAMES_KK`. Those are
English letter names for acronyms (`H` is «эйч», `C` is «си») and are wrong for
chemistry. A new table is required.

### Disambiguation rule

Physics variables in the corpus look identical to formulas: `U1 U2 V1 V2 F1 F2
S1 S2 I2 N0 B0 K1 C1 W2`. Each of those letters is also a real element symbol
(uranium, vanadium, fluorine, sulfur, iodine, boron, potassium, carbon,
tungsten). Reading `U2` as «уран два» would be a regression.

**Rule — count element symbols in the token:**

- exactly **one** symbol: read as variable with index using the *physics letter
  table*. `U1` is «у один», `V2` is «вэ два».
- **two or more** symbols: read as a formula using the *chemistry* names.
  `KOH` is «калий о аш», `H2O` is «аш два о», `KMnO4` is «калий марганец о
  четыре».

The two branches agree where it matters: `H2` «аш два», `O2` «о два», `N2` «эн
два» are correct either way.

### Reaction equations

`_protect_nonlinguistic` currently runs `_EQUATION_RE` and
`_MATH_EXPRESSION_RE` **before** `_protect_chemical_formulas`, so
`2H₂ + O₂ → 2H₂O` is swallowed as a math expression and restored raw.

Add a reaction protection pass **ahead of** those two patterns, matching two or
more formula tokens joined by `+` or an arrow, and expand the whole span with
`speak_reaction`. If any participant fails the formula check, the pass declines
and leaves the span for the existing math handling.

Target output:

```
2H₂ + O₂ → 2H₂O
RU: два аш два плюс о два образуется два аш два о
KK: екі аш екі плюс о екі түзіледі екі аш екі о
```

### Letter tables

Chemistry one-letter symbols (both languages, identical):

```
H аш   B бор   C це   N эн   O о    F фтор  P пэ
S эс   K калий V ванадий  I йод  W вольфрам  U уран  Y иттрий
```

Physics letter names, single-symbol branch (Russian tradition, both languages):

```
A а    B бэ   C цэ   D дэ   E е    F эф   G гэ   H аш   I и
J жи   K ка   L эль  M эм   N эн   O о    P пэ   Q ку   R эр
S эс   T тэ   U у    V вэ   W дубль-вэ    X икс  Y игрек Z зет
```

Two-letter element names: full periodic table with standard Russian
Latin-derived names. School-critical entries fixed here to prevent drift:

```
Na натрий  Mg магний   Al алюминий  Si силиций  Cl хлор    Ca кальций
Fe феррум  Cu купрум   Zn цинк      Ag аргентум Ba барий   Pb плюмбум
Hg гидраргирум  Mn марганец  Cr хром  Ni никель  Co кобальт Sn станнум
Br бром    Li литий    Be бериллий  He гелий    Ne неон    Ar аргон
Ti титан   Sr стронций Se селен     Au аурум    Pt платина Cd кадмий
```

Kazakh uses the same element and letter tables; only the numerals
(`екі`, `үш`, `төрт`), the multiplier form (`рет`) and the arrow word
(`түзіледі`) differ.

## Files

```
NEW  voice/app/tts/formula_speech.py
NEW  voice_omnivoice/app/formula_speech.py
EDIT voice/app/tts/abbreviation_normalization.py        hook + reaction pass
EDIT voice_omnivoice/app/abbreviation_normalization.py  hook + reaction pass
NEW  tests/test_chemical_formula_speech.py
EDIT tests/test_tts_text_normalization_backends.py
```

## Testing

`pytest`, fully mocked, no GPU. Cases per language:

- `H2O`, `CO2`, `CO₂`, `NaCl`, `NaOH`, `HCl`, `H2SO4`, `CuSO4`, `CaCO3`,
  `NaHCO3`, `KMnO4`, `AgNO3`, `C6H12O6`, `Ca(OH)2`, `Al2(SO4)3`
- regression guards: `U1`, `U2`, `V1`, `F2`, `S1`, `B0`, `K1` read as variables
- pass-through guards: `OKULYK`, `KWWSV`, `FF12`, `ISBN`, `pH`, `DNA` unchanged
- charges: `Ca2+`, `Ca²⁺`, `SO4²⁻`
- reaction: `2H₂ + O₂ → 2H₂O`
- end-to-end through `normalize_russian_tts_text` /
  `normalize_kazakh_tts_text`, asserting no bare Latin remains

## Out of scope

The KK-MMS fallback inside the `voice` container has no normalization at all
(no numbers, no abbreviations, and now no formulas). `omnivoice` is the Kazakh
default; add normalization there if MMS ever becomes primary.

## Delivery

After tests pass: render samples locally from the deployed `/tts` for RU
(supertonic and qwen) and KK (omnivoice), then deploy to
`megroup-b560m-hdv-m-2`.
