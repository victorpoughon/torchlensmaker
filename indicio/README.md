# indicio

`indicio` is a python package that provides offline access to the
[refractiveindex.info](https://refractiveindex.info) database of optical
constants. Contrary to other similar packages, the full upstream database ships
inside the package, so there is no need to download it separately: just `pip
install indicio` and you are ready to go.

`indicio` has zero runtime dependency, not even YAML because the database is
shipped in a custom sqlite format that python can decode natively. This also
reduces the size of the database to ~16MB.

`indicio` returns **structured descriptions of dispersion models**. It does
implement formulas to compute `n(λ)` or `k(λ)`. That job is left to the
consumer, who can pick whatever numerical stack suits the use case (numpy, jax,
torch, etc.).

## Installation

```bash
pip install indicio
```

## Quickstart

```python
import indicio

SiO2 = indicio.get_material("main", "SiO2", "Malitson")
print(SiO2.n)

>>> Sellmeier(c1=0.0, coefficients=((0.6961662769317627, 0.06840430200099945), (0.40794259309768677, 0.11624140292406082), (0.8974794149398804, 9.896161079406738)), wavelength_range=(0.21, 6.7))
```

## Conventions

- **Wavelengths are micrometers everywhere**, matching the convention of
  [refractiveindex.info](https://refractiveindex.info).
- **Tabulated arrays are raw bytes**, packed as little-endian IEEE 754
  float32. Decode with numpy or with the standard library `array` module
  (recipe below).
- **One model per quantity per material.** A material has at most one
  description for `n` and one for `k`; either may be `None`.

## Lookup and browsing API

```python
indicio.get_material(shelf, book, page) -> MaterialEntry
indicio.has_material(shelf, book, page) -> bool
indicio.shelves() -> tuple[str, ...]
indicio.books(shelf) -> tuple[str, ...]
indicio.pages(shelf, book) -> tuple[str, ...]
indicio.iter_materials() -> Iterator[MaterialEntry]
indicio.search(query) -> tuple[tuple[str, str, str], ...]

indicio.__version__               # library version, e.g. "1.0.0"
indicio.__database_version__      # upstream commit hash baked into the data
```

## The data model

```python
@dataclass(frozen=True)
class MaterialEntry:
    shelf: str
    book: str
    page: str
    name: str | None
    references: str | None
    comments: str | None
    n: NModel | None              # real part — None if upstream has only k
    k: KModel | None              # imaginary part — None if upstream has only n
```

For detailed data description of each model see
[src/indicio/models.py](src/indicio/models.py).

Models are either tabulated or closed-form formulas, following the upstream
[refractiveindex.info dispersion-formula spec](https://refractiveindex.info/database/doc/Dispersion%20formulas.pdf).

## Examples

The `examples/` directory contains complete evaluator scripts covering every
formula type, including pattern-match dispatch on `entry.n` / `entry.k`.

- [`examples/example_evaluation_stdlib.py`](examples/example_evaluation_stdlib.py)
  — uses only the standard library (`math`, `array`, `bisect`).
- [`examples/example_evaluation_numpy.py`](examples/example_evaluation_numpy.py)
  — uses numpy; evaluators are vectorized over a wavelength array.
- [`examples/example_plot.py`](examples/example_plot.py) — uses numpy +
  matplotlib to plot n(λ) and k(λ) over the validity range.

```bash
uv run python examples/example_evaluation_stdlib.py main SiO2 Malitson
uv run python examples/example_evaluation_numpy.py main Au Johnson
uv run python examples/example_plot.py main BaF2 Bosomworth-300K
```

### Numeric precision

All numeric data is stored as float32. This gives ~7 significant decimal
digits, which exceeds the precision of the source YAML (typically 4–6
digits) and halves the size of tabulated payloads. Consumers who need
float64 should cast at evaluation time.

## Versioning

`indicio` follows semantic versioning for its API. The bundled database is
identified independently:

```python
indicio.__version__               # API/library version
indicio.__database_version__      # upstream refractiveindex.info commit hash
```

API-breaking changes to the dataclasses bump the major version.
Database-only updates bump the patch version.

## Licensing and attribution

The refractiveindex.info database is distributed under
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) (public
domain). When publishing results that rely on a particular material entry,
cite the original authors — their citation strings are preserved verbatim
in `MaterialEntry.references`.

```python
print(indicio.get_material("main", "SiO2", "Malitson").references)
```

The `indicio` library code itself is distributed under its own LICENSE in
the source tree.
