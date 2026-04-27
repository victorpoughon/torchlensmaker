# `indicio` — Library Design Document

## Overview

A Python library providing offline access to the [refractiveindex.info](https://refractiveindex.info) database. The library ships the entire database as part of the package, so users only need to `pip install indicio` to get going — no manual YAML downloads, no first-run network fetches, no environment variables pointing at a data directory.

The library's scope is intentionally narrow: it returns **structured descriptions of dispersion models** for each material entry. It does *not* compute refractive index values as a function of wavelength. That responsibility is delegated to consumers, who can use whatever numerical stack they prefer (numpy, jax, torch, plain Python, symbolic libraries, etc.).

This design choice has two important consequences:

1. The library has **no required runtime dependencies** beyond the Python standard library.
2. The library exposes a clean, typed, easily testable data model that is decoupled from any particular evaluation strategy.

## Goals

- Zero-configuration install: `pip install indicio` and import.
- Fully offline: no network access required at install or runtime.
- Small footprint: target under 15 MB installed (down from ~45 MB of raw YAML).
- Fast lookups: O(1) access to any material by `(shelf, book, page)`.
- Stdlib-only runtime: no numpy, no scipy, no PyYAML at runtime.
- Strong typing: every dispersion model is a typed dataclass; users get full IDE and type-checker support.
- Faithful representation: preserve all metadata, including per-entry citations and validity ranges.

## Non-goals

- **Computing n(λ) or k(λ).** This is left to consumers. The library returns model descriptions, not values.
- **Plotting, fitting, or unit conversion utilities.** Out of scope.
- **Mutating or extending the database at runtime.** The shipped database is read-only. Users who need custom materials can build their own data structures using the same dataclasses.
- **Tracking the live upstream database.** Updates ship via new library releases.

## Architecture

### Package layout

```
indicio/
├── __init__.py          # Public API re-exports
├── models.py            # Dispersion model dataclasses
├── database.py          # Lookup, browsing, and query logic
├── _loader.py           # Internal SQLite access
├── data/
│   └── refractiveindex.db   # Bundled SQLite database
└── py.typed             # PEP 561 marker
```

### Data storage

The database is shipped as a single **SQLite file** embedded in the package via `package_data` / `MANIFEST.in`. SQLite was chosen because:

- It is part of the Python standard library (`sqlite3`), keeping runtime dependencies at zero.
- It supports indexed lookups by `(shelf, book, page)` — effectively O(1) material access without parsing thousands of YAML files at import time.
- It compresses tabulated numeric data well when stored as binary blobs.
- It is a single file, simple to bundle and reason about.

Tabulated (n, k) data is stored as `BLOB` columns containing packed `float64` arrays in native byte order. Float64 is used uniformly throughout the database: the simplification of having a single precision is worth the modest size cost over a mixed float32/float64 scheme, and float64 eliminates any concern about losing precision relative to the source YAML. Decoding uses `struct` or `array` from the standard library — no numpy required.

Estimated final size: **10–15 MB** for the bundled database. Optional zlib compression on the blobs (using stdlib `zlib`) can claw back several MB if needed, at the cost of a small per-lookup decompression step.

### Build pipeline

A separate `scripts/build_database.py` (not shipped in the wheel) is responsible for:

1. Pulling a pinned commit of [`polyanskiy/refractiveindex.info-database`](https://github.com/polyanskiy/refractiveindex.info-database).
2. Parsing every YAML entry (PyYAML is a *build-time* dependency only).
3. Converting each entry to the appropriate dataclass.
4. Serializing into the SQLite schema.
5. Validating round-trip equality on a sample of entries.

The pinned upstream commit hash is recorded in the package as `indicio.__database_version__` so users can identify exactly what data they have.

### Schema

```sql
CREATE TABLE materials (
    shelf       TEXT NOT NULL,
    book        TEXT NOT NULL,
    page        TEXT NOT NULL,
    name        TEXT,
    references_ TEXT,    -- citation text, preserved verbatim
    comments    TEXT,
    PRIMARY KEY (shelf, book, page)
);

CREATE TABLE model_pieces (
    shelf       TEXT NOT NULL,
    book        TEXT NOT NULL,
    page        TEXT NOT NULL,
    quantity    TEXT NOT NULL,        -- 'n' or 'k'
    piece_idx   INTEGER NOT NULL,     -- order within the quantity
    model_kind  TEXT NOT NULL,        -- discriminator: 'sellmeier', 'cauchy', 'tabulated', ...
    payload     BLOB NOT NULL,        -- model-specific serialized fields
    wl_min      REAL,
    wl_max      REAL,
    PRIMARY KEY (shelf, book, page, quantity, piece_idx),
    FOREIGN KEY (shelf, book, page) REFERENCES materials(shelf, book, page)
);

CREATE INDEX idx_book ON materials(book);
```

A single material entry typically has one piece for `n` and one for `k`, but either side may have multiple pieces (a piecewise description across wavelength windows) or be empty (when the upstream entry only describes one of the two quantities). Combined `tabulated nk` blocks from upstream are split by the build script into one `n` piece and one `k` piece.

## Public API

### Dispersion model dataclasses

All dataclasses are frozen and live in `indicio.models`. Wavelengths throughout the entire library — in dataclass fields, return values, and documentation — are expressed in **micrometers**, matching the upstream convention. The library performs no unit conversion; consumers working in nanometers or meters apply the trivial scaling themselves.

Each model carries its own validity range as a `WavelengthRange` instance.

#### Encoding of tabulated arrays

Tabulated entries store their numeric arrays as **raw `bytes`** rather than as Python tuples or lists. This is the most efficient way to expose the data when consumers will likely feed it directly to numpy or another array library, and it avoids materializing potentially-large lists of Python floats just to throw them away.

The binary layout is fixed: **packed IEEE 754 float64 values in native byte order**. This is documented in the dataclass docstrings and is part of the API contract — consumers can rely on it without inspecting any per-instance metadata.

Consumers decode the bytes with whichever tool they prefer:

```python
# With numpy:
import numpy as np
wls = np.frombuffer(piece.wavelength_um, dtype=np.float64)

# Without numpy, using stdlib only:
import array
wls = array.array("d")
wls.frombytes(piece.wavelength_um)
```

Each tabulated dataclass also exposes a `length` field giving the number of points, so consumers can sanity-check the buffer size without computing it.

#### Dataclass definitions

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class WavelengthRange:
    min_um: float
    max_um: float

@dataclass(frozen=True)
class TabulatedN:
    """Tabulated real refractive index n(λ). Linear interpolation is the
    upstream convention but the library does not perform it.

    `wavelength_um` and `n` are raw bytes containing `length` packed
    float64 values each, in native byte order."""
    wavelength_um: bytes
    n: bytes
    length: int
    wavelength_range: WavelengthRange

@dataclass(frozen=True)
class TabulatedK:
    wavelength_um: bytes
    k: bytes
    length: int
    wavelength_range: WavelengthRange

@dataclass(frozen=True)
class Sellmeier:
    """Standard Sellmeier: n²-1 = Σ Bᵢ λ² / (λ² - Cᵢ)."""
    coefficients: tuple[tuple[float, float], ...]   # list of (B, C) pairs
    wavelength_range: WavelengthRange

@dataclass(frozen=True)
class Sellmeier2:
    """Alternative Sellmeier form used by refractiveindex.info (formula 2)."""
    coefficients: tuple[tuple[float, float], ...]
    wavelength_range: WavelengthRange

@dataclass(frozen=True)
class Polynomial:
    """n² = c₀ + Σ cᵢ λ^(eᵢ)."""
    terms: tuple[tuple[float, float], ...]   # list of (coefficient, exponent)
    wavelength_range: WavelengthRange

@dataclass(frozen=True)
class RefractiveIndexInfoFormula:
    """Generic container for the numbered formulas (1 through 9) defined
    by refractiveindex.info, when no more specific dataclass applies."""
    formula_number: int
    coefficients: tuple[float, ...]
    wavelength_range: WavelengthRange

@dataclass(frozen=True)
class Cauchy:
    """n = A + B/λ² + C/λ⁴ + ..."""
    coefficients: tuple[float, ...]   # [A, B, C, ...]
    wavelength_range: WavelengthRange

@dataclass(frozen=True)
class Gases:
    """Gas formula: n-1 = Σ Bᵢ / (Cᵢ - λ⁻²)."""
    coefficients: tuple[tuple[float, float], ...]
    wavelength_range: WavelengthRange

# ... one dataclass per upstream formula type ...

# Models that describe the real part n(λ).
NModel = (
    TabulatedN
    | Sellmeier | Sellmeier2 | Polynomial
    | Cauchy | Gases | RefractiveIndexInfoFormula
)

# Models that describe the imaginary part k(λ). In practice k is almost
# always tabulated upstream, but the union allows for formula-based k
# entries without an API change.
KModel = (
    TabulatedK
    | RefractiveIndexInfoFormula
)
```

### Material entries

A material entry separates the description of the real part `n(λ)` from the imaginary part `k(λ)`. Each is held as a tuple of model pieces — one element in the common case, more if the upstream entry describes the quantity piecewise across multiple wavelength windows.

```python
@dataclass(frozen=True)
class MaterialEntry:
    shelf: str
    book: str
    page: str
    name: str | None
    references: str | None
    comments: str | None
    n: tuple[NModel, ...]   # may be empty if upstream provides only k
    k: tuple[KModel, ...]   # may be empty if upstream provides only n
```

#### Why separate `n` and `k` rather than a flat list of segments?

The upstream YAML format permits a list of `DATA` entries per material, and a single material commonly contains both an `n` description (often a formula) and a `k` description (usually tabulated). A flat list conflates two distinct concerns:

- **Different quantities** (n vs k) — these are complementary, not alternatives. Consumers almost always want to filter by quantity before doing anything else.
- **Different wavelength windows for the same quantity** — a genuinely piecewise description, which is rare but real.

If the library exposed a flat `segments: tuple[...]` field, every consumer would have to write the same filter logic to pick out "the n part" or "the k part." Splitting `n` and `k` at the API level removes that boilerplate and makes intent explicit. The piecewise-window case is still handled naturally by the tuple having more than one element.

#### Handling combined nk tabulated entries

The upstream database has a `tabulated nk` block type — a single table containing wavelengths, n values, and k values together. The build script **splits these into separate `TabulatedN` and `TabulatedK` instances** when populating the bundled SQLite database. The wavelength array is duplicated in both, which costs a small amount of disk space but keeps the consumer-facing model uniform: `entry.n` always contains only n descriptions, `entry.k` always contains only k descriptions, and consumers never have to special-case a combined type.

#### Example

Consumers pattern-match on each model to decide what to do:

```python
import math
import array
from indicio import get_material
from indicio.models import Sellmeier, TabulatedN, TabulatedK

entry = get_material("main", "SiO2", "Malitson")

# Build an evaluator for the real part:
for piece in entry.n:
    match piece:
        case Sellmeier(coefficients=coeffs, wavelength_range=wr):
            def n(wavelength_um: float) -> float:
                return math.sqrt(1 + sum(
                    B * wavelength_um**2 / (wavelength_um**2 - C)
                    for B, C in coeffs
                ))
        case TabulatedN(wavelength_um=wl_bytes, n=n_bytes, length=n_pts):
            wls = array.array("d"); wls.frombytes(wl_bytes)
            ns  = array.array("d"); ns.frombytes(n_bytes)
            # ...consumer interpolates however they like.

# Build an evaluator for the imaginary part if one is provided:
for piece in entry.k:
    match piece:
        case TabulatedK(wavelength_um=wl_bytes, k=k_bytes, length=n_pts):
            ...
```

This is the central design decision of the library. The library tells you *what the model is*; you decide *how to evaluate it*.

### Lookup and browsing

```python
def get_material(shelf: str, book: str, page: str) -> MaterialEntry: ...

def has_material(shelf: str, book: str, page: str) -> bool: ...

def shelves() -> tuple[str, ...]: ...

def books(shelf: str) -> tuple[str, ...]: ...

def pages(shelf: str, book: str) -> tuple[str, ...]: ...

def iter_materials() -> Iterator[MaterialEntry]:
    """Stream every material entry. Useful for building search indices
    or exporting to other formats."""

def search(query: str) -> tuple[tuple[str, str, str], ...]:
    """Simple substring search over book and page names. Returns
    (shelf, book, page) triples."""
```

### Top-level constants

```python
__version__: str             # library version
__database_version__: str    # upstream commit hash baked into the data
```

## Why no computation?

This deserves justification because it is unusual for libraries in this space.

**Numerical opinions are not the library's to make.** The upstream database documents formulas; it does not specify a single evaluation policy. Real consumers have legitimate, conflicting needs: some want vectorized numpy, some want jax for autodiff, some want symbolic expressions for code generation, some want strict bounds checking, some want extrapolation, some want cubic interpolation rather than linear. A library that picks one policy frustrates everyone else.

**Decoupling improves testability.** The library can be tested for data integrity (does the dataclass round-trip the YAML?) without ever running a numerical computation. Consumers can test their evaluator logic against synthetic dataclasses without touching the database.

**It eliminates the numpy dependency.** With no array math at runtime, the package depends only on the standard library. This matters more than it sounds: it makes the package trivial to vendor, trivial to use in restricted environments (AWS Lambda, embedded Python, MicroPython-adjacent setups), and immune to numpy ABI churn.

**It encourages clean downstream code.** A consumer who writes their own evaluator ends up with a small, explicit function they understand, rather than relying on hidden behavior in a third-party library.

The cost is that simple use cases ("just give me n at 633 nm") require a few extra lines from the user. The library should mitigate this by including, in its documentation, a small recipes page with copy-pasteable evaluator implementations for each model type. This is documentation, not code shipped in the package.

## Licensing and attribution

The refractiveindex.info database is distributed under [CC-BY](https://creativecommons.org/licenses/by/4.0/) for the database structure, with individual entries attributed to their original authors. Shipping the database imposes obligations:

- The package includes the upstream `LICENSE` and a `NOTICE` file pointing at the source repository and the pinned commit.
- Per-entry `references` strings from the YAML are preserved verbatim in `MaterialEntry.references`, so users can cite original authors correctly.
- The README prominently directs users to cite both the library and the underlying data sources.

## Versioning

The library follows semantic versioning for its API. The `__database_version__` field tracks the upstream commit independently. A typical release commit message reads:

```
indicio 1.4.0 — bump database to upstream commit a1b2c3d
```

API-breaking changes to the dataclasses bump the major version. Database-only updates bump the patch version.

## Open questions

- **Search ergonomics.** The proposed `search()` is deliberately minimal. A future extension could add fuzzy matching or fielded queries, but only if real usage demonstrates a need.
- **Data-only companion package.** If the bundled database grows substantially in future upstream revisions, splitting into `indicio` (code) and `indicio-data` (data) becomes attractive. The API would not change; only the install topology.

## Summary

`indicio` is a small, focused library: it ships the entire refractiveindex.info database in a compact SQLite file, exposes every dispersion model as a typed dataclass, and stops there. It has no runtime dependencies outside the standard library. It computes nothing. Consumers get clean structured data and full freedom over how to use it.
