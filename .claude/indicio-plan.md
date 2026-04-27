# `indicio` — Implementation Plan

This plan implements the design described in `.claude/indicio.md`. Each step is small and verifiable on its own. Steps are ordered so we can stop after step 1 and look at the resulting database file size before committing to the rest.

## Working layout

```
indicio/
├── pyproject.toml
├── refractiveindex.info-database/   # upstream clone (build-time only)
├── scripts/                         # build tooling, NOT shipped in the wheel
│   └── build_database.py
└── src/indicio/
    ├── __init__.py
    ├── models.py
    ├── database.py
    ├── _loader.py
    ├── data/
    │   └── refractiveindex.db
    └── py.typed
```

The `scripts/` directory and `refractiveindex.info-database/` clone are kept outside `src/` so they are never accidentally installed.

---

## Step 1 — YAML → SQLite conversion script

**Goal:** produce `src/indicio/data/refractiveindex.db` from the upstream clone, so we can measure the resulting file size before going further.

This step deliberately limits its scope to data conversion. It does **not** touch the public API yet — the dataclasses defined here are private build-script helpers; the public dataclasses arrive in step 3 and may differ in shape.

### 1.1 — Survey the upstream `DATA` block types ✅ DONE

`scripts/survey_types.py` walks every `*.yml` and tallies `DATA.type` values. Findings (4,164 files, 0 parse failures):

- **Quantities present:** `tabulated n/k/nk/n2`, `formula 1` through `formula 9`.
- **Block shapes:** every file has 1 or 2 blocks — no piecewise multi-window descriptions exist in the current upstream dataset. Top shape is `('formula 3', 'tabulated k')` (972 files).
- **460 "no DATA" files** are all `about.yml` documentation stubs (glass-family text descriptions). They are not referenced by the catalogues, so a catalogue-driven walk skips them naturally.

Decisions confirmed:

1. **Skip `tabulated n2` entirely** for v0.1 (Kerr nonlinear index, different physics from n/k). Logged as a single summary warning at build time. Listed under "Out of scope" below.
2. **Drive the build off `catalog-nk.yml`**, not a recursive glob — sidesteps the `about.yml` files cleanly.
3. **Land all formulas as generic `formula_N` rows** in step 1.2; map them to specific dataclasses (Sellmeier, Cauchy, …) in step 3 once the dispersion-formulas PDF has been digested. This keeps the build script free of formula-taxonomy work and means we can iterate on the mapping without rebuilding the DB.

### 1.2 — Implement `scripts/build_database.py`

Single-file script. Build-time deps: `pyyaml`. No runtime deps.

Responsibilities:

1. Locate the upstream clone (default: `../refractiveindex.info-database` relative to the script — already in `indicio/refractiveindex.info-database/`).
2. Record the upstream commit hash (`git -C <clone> rev-parse HEAD`) for embedding into the package later.
3. Walk `catalog-nk.yml` to enumerate `(shelf, book, page) → data file` plus the human-readable `name`. Catalogues also carry `DIVIDER` entries — those are display hints and can be discarded. **`catalog-n2.yml` is skipped entirely** (n2 out of scope for v0.1). The catalog has 17 exact-duplicate entries (same key, same data file, listed under two DIVIDERs) which are silently collapsed; if upstream ever introduces a duplicate with conflicting paths the build aborts.
4. For each material:
   - Load the YAML data file.
   - Extract `REFERENCES` (interned via the `refs` table — one row per unique citation string) and `COMMENTS` (stored verbatim per material).
   - Convert each `DATA` block to a `(quantity, model_kind, payload_bytes, n_points, wl_min, wl_max)` row, splitting `tabulated nk` blocks into one `n` row and one `k` row sharing the wavelength array.
   - At most one `n` row and one `k` row per material. The survey identified one upstream file (`organic/.../polyvinylpyrrolidone/nk/Konig.yml`) that supplies both a formula and a tabulated representation of the same `n`; the build script logs and applies "first wins" (formula kept). All other potential collisions abort.
5. Write everything into a fresh SQLite file using the schema from the design doc. Open with `journal_mode=OFF`, `synchronous=OFF` and one transaction around the whole insert for speed.
6. After insertion, run `VACUUM` to tighten file size, then print the byte size.

### 1.3 — Payload encoding (build-side helpers)

For each model kind, define a small encode function. The payload is consumed only by the loader in step 4, so the format is internal — the only constraint is round-trippability and stdlib-decodability.

All numeric data is packed as **little-endian IEEE 754 float32**. Float32 gives ~7 significant decimal digits, which exceeds the precision of the source YAML (typically 4–6 digits) and halves tabulated payload size relative to float64.

Tabulated payloads are additionally **zlib-compressed at level 9** before being stored. Formula payloads are stored raw (a handful of coefficients each — compression overhead would dwarf any gain). The discriminator column `model_kind` tells the loader whether to inflate.

Encoding plan (option A — generic `formula_N` rows, refined into typed dataclasses at load time in step 4):

| `model_kind`    | Payload layout                                                                  |
|-----------------|---------------------------------------------------------------------------------|
| `tabulated_n`   | zlib(concat of `wavelength_um` blob + `n` blob), `n_points` set                 |
| `tabulated_k`   | zlib(concat of `wavelength_um` blob + `k` blob), `n_points` set                 |
| `formula_1`–`formula_9` | packed float32 array of raw upstream coefficients, no compression       |

`tabulated_n2` is dropped from the input (not encoded).

The build script does **not** know which formula number corresponds to Sellmeier vs Cauchy vs Gases — it only preserves the raw coefficient list. Mapping each `formula_N` to a typed dataclass (Sellmeier, Sellmeier2, Cauchy, Gases, …) happens in step 3 (defining dataclasses) and step 4 (decode in `_loader.py`) per the upstream "Dispersion formulas" doc shipped at `database/doc/Dispersion formulas.pdf`. The benefit: getting the formula taxonomy wrong only requires reshipping `_loader.py`, not regenerating the bundled DB.

### 1.4 — Smoke checks built into the script

- Total material count printed at the end matches the catalogue count.
- For one well-known entry (e.g. `main / SiO2 / Malitson`), pretty-print the rows so we can eyeball that coefficients survive the round trip.
- Final on-disk size of the `.db` file is printed, plus a payload-size breakdown by `model_kind`.

### 1.5 — Run it ✅ DONE

`uv run python scripts/build_database.py`. Final size after VACUUM: **15.24 MiB**, on target.

---

## Step 2 — Lock in storage decisions ✅ DONE

Three iterations measured during step 1:

| format | DB size after VACUUM |
|---|---|
| float64, no compression | 36.49 MiB |
| float32, zlib on tabulated only | 15.89 MiB |
| float32, zlib on tabulated, interned `refs` table | **15.24 MiB** |

Decisions locked in (and reflected in `indicio.md`):

- **Float32 little-endian** for every numeric value in payloads. Source YAML rarely has more than 5 significant figures; float32's ~7 digits is plenty.
- **Tabulated payloads only are zlib(level 9)-compressed.** Formula payloads stored raw — they are tiny and compression overhead would dwarf gains. The `model_kind` discriminator already tells the loader whether to inflate, so no extra `compressed` flag column is needed.
- **References interned in a separate `refs` table.** 3,543 materials → 721 unique citation strings (the dominant repeat is the OHARA Zemax catalog citation, ×413). Interning saves ~0.65 MiB and keeps the public `MaterialEntry.references` field unchanged thanks to a join in the loader.
- **Pin the upstream commit hash** by writing it to the `meta` table (already done) and generating `src/indicio/_database_version.py` at build time so it can be imported as `indicio.__database_version__`.

No further iteration on the storage format unless future upstream growth pushes us back over budget.

---

## Step 3 — Public dataclasses (`src/indicio/models.py`)

Translate the design doc's dataclass section verbatim:

- `WavelengthRange`
- `TabulatedN`, `TabulatedK` (with `wavelength_um`, `n`/`k`, `length`, `wavelength_range`)
- `Sellmeier`, `Sellmeier2`, `Polynomial`, `Cauchy`, `Gases`, `RefractiveIndexInfoFormula`
- `MaterialEntry` with `n: NModel | None` and `k: KModel | None` (Optional, not tuple)
- Type aliases `NModel`, `KModel`

All `frozen=True`. No methods beyond what `@dataclass` generates. Module has zero non-stdlib imports.

Add `from __future__ import annotations` to keep forward refs cheap.

This is also where the `formula_N → specific dataclass` mapping is finalised, using `database/doc/Dispersion formulas.pdf` as the authoritative reference. The mapping itself lives in step 4's loader; this step just defines the targets.

---

## Step 4 — Internal loader (`src/indicio/_loader.py`)

A thin layer over `sqlite3` that:

- Resolves the bundled DB via `importlib.resources.files("indicio.data") / "refractiveindex.db"`.
- Opens the connection in read-only mode (`uri=True`, `mode=ro`).
- Caches the connection at module level (lazy init).
- Provides `fetch_material(shelf, book, page)`, `iter_pks()`, `list_shelves()`, `list_books(shelf)`, `list_pages(shelf, book)`, `search_substring(query)`.
- Decodes a row → `MaterialEntry` by dispatching on `model_kind` and reconstructing the right dataclass with `bytes`/`tuple` payloads.

The decode functions are the inverse of the encoders in step 1.3. They are tested directly in step 6.

---

## Step 5 — Public API surface (`src/indicio/database.py`, `src/indicio/__init__.py`)

`database.py`: thin wrappers around `_loader` matching the design doc:

```python
def get_material(shelf, book, page) -> MaterialEntry
def has_material(shelf, book, page) -> bool
def shelves() -> tuple[str, ...]
def books(shelf) -> tuple[str, ...]
def pages(shelf, book) -> tuple[str, ...]
def iter_materials() -> Iterator[MaterialEntry]
def search(query) -> tuple[tuple[str, str, str], ...]
```

`__init__.py`: re-export the public dataclasses, the lookup functions, plus `__version__` (read from `importlib.metadata`) and `__database_version__` (imported from the generated `_database_version` module).

---

## Step 6 — Tests

Use pytest. All run with `uv run pytest`.

1. **Schema integrity:** every row in `model_pieces` has a known `model_kind`; every `(shelf, book, page)` referenced in `model_pieces` exists in `materials`.
2. **Counts:** number of materials equals what the catalogue parser sees.
3. **Round-trip on a curated sample** (~10 entries chosen to cover every model kind): re-parse the source YAML, build the in-memory dataclass via the build helpers, fetch the same key from the bundled DB, and assert structural equality. Tabulated arrays compared after `array.array` decode.
4. **Numeric spot checks:** for famous entries (`main/SiO2/Malitson` Sellmeier; `main/Au/Johnson` tabulated), assert the first/last coefficient matches a hard-coded expected value.
5. **API ergonomics:** `get_material` raises a `KeyError` with a useful message for unknown keys; `has_material` returns `False`. `shelves()/books()/pages()` are sorted and de-duplicated.
6. **Read-only:** opening a second connection write-mode and inserting fails (sanity check that the loader's read-only flag works).
7. **No numpy:** import the package in a subprocess with `numpy` blocked from `sys.modules` and confirm it works.

---

## Step 7 — Packaging

- Add `src/indicio/data/refractiveindex.db` to the wheel. With `uv_build`, place it under `src/indicio/data/` and ensure `package-data` / `force-include` settings pick up `*.db` and `py.typed`.
- Ship `LICENSE` (upstream CC0/CC-BY) and a `NOTICE` pointing at the upstream commit.
- Smoke test: `uv build`, `pip install dist/indicio-*.whl` into a throwaway venv, `python -c "import indicio; print(indicio.get_material('main','SiO2','Malitson'))"`.

---

## Step 8 — Build pipeline reproducibility

- Document the exact upstream commit in `scripts/build_database.py` (constant at top of file). The script refuses to run if `git -C <clone> rev-parse HEAD` does not match — prevents silent drift.
- Add a `scripts/README.md` explaining the build flow for future maintainers.

---

## Step 9 — Documentation

- README with a 30-line quickstart and a recipes section: copy-pasteable evaluators for `Sellmeier`, `TabulatedN` (with and without numpy), and `Cauchy`.
- Reiterate that the library does not compute n(λ) — link the recipes.

---

## Out of scope for v0.1

- Fuzzy search, fielded queries.
- `tabulated n2` support (Kerr nonlinear index — different physics; revisit when there's a concrete consumer).
- Piecewise multi-window descriptions per quantity (not present in current upstream; would require a major version bump from `n: NModel | None` to `n: tuple[NModel, ...]`).
- Splitting into `indicio` + `indicio-data` packages.
- Build-time CLI for users to regenerate the DB from a different upstream commit.

These are listed in the design doc's "Open questions" / "Non-goals" and are deferred until concrete need.

---

## Stop-and-look checkpoints

- **After 1.5:** verify DB size is in the 10–15 MB ballpark before committing to the rest of the plan.
- **After step 4:** confirm a single round-trip works end-to-end on one entry before fleshing out the public API.
- **After step 6:** run the full test suite before packaging.
