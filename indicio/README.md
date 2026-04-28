# indicio

`indicio` is a python package that provides offline access to the
[refractiveindex.info](https://refractiveindex.info) database of optical
constants. Contrary to other similar packages, the full upstream database ships
inside the package, so there is no need to download it separately: just `pip
install indicio` and you are ready to go.

`indicio` has zero runtime dependency, not even YAML because the database is
stored internally in a custom sqlite format that python can decode natively.
This also reduces the size of the database to ~16MB.

`indicio` returns structured descriptions of dispersion models. It does
**not** implement formulas to compute `n(λ)` or `k(λ)`. That job is left to the
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

## API

```python
indicio.get_material(shelf: str, book: str, page: str) -> MaterialEntry
indicio.has_material(shelf: str, book: str, page: str) -> bool
indicio.shelves() -> tuple[str, ...]
indicio.books(shelf: str) -> tuple[str, ...]
indicio.pages(shelf: str, book: str) -> tuple[str, ...]
indicio.iter_materials() -> Iterator[MaterialEntry]
indicio.search(query: str) -> tuple[tuple[str, str, str], ...]

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

* For the dispersion model n, there are 10 possible cases: 9 closed-form
  formulas (following the upstream [dispersion-formula
  doc](https://refractiveindex.info/database/doc/Dispersion%20formulas.pdf)), or
  tabulated data.

* For the extinction coefficient k, the data is always tabulated.

For detailed data description of each model see
[src/indicio/models.py](src/indicio/models.py). 


## Examples

The `examples/` directory contains complete evaluation scripts covering every
formula type:

- [`examples/example_evaluation_stdlib.py`](examples/example_evaluation_stdlib.py): uses only the standard library (`math`, `array`, `bisect`).
- [`examples/example_evaluation_numpy.py`](examples/example_evaluation_numpy.py): uses numpy; evaluators are vectorized over a wavelength array.
- [`examples/example_plot.py`](examples/example_plot.py): uses numpy +
  matplotlib to plot n(λ) and k(λ) over the validity range.

```bash
python examples/example_evaluation_stdlib.py main SiO2 Malitson
python examples/example_evaluation_numpy.py main Au Johnson
python examples/example_plot.py main BaF2 Bosomworth-300K
```

## Conventions

- Wavelengths are micrometers everywhere, matching the convention of
  [refractiveindex.info](https://refractiveindex.info).
- Tabulated arrays are raw bytes, packed as little-endian IEEE 754
  float32. See examples above for decoding.

## Licensing and attribution

The refractiveindex.info database is distributed under [CC0
1.0](https://creativecommons.org/publicdomain/zero/1.0/) (public domain). When
publishing results that rely on a particular material entry, please cite the
original authors. Their citation strings are preserved verbatim in
`MaterialEntry.references`.

```python
print(indicio.get_material("main", "SiO2", "Malitson").references)
```

The `indicio` library code itself is distributed under its own LICENSE in
the source tree.
