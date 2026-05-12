# Trama Export Support Plan

This document is a step-by-step plan for adding `trama` export (and round-trip
import) support to the user-facing modules of `torchlensmaker`. It assumes
`trama>=0.2.0` is already declared as a dependency in
`torchlensmaker/pyproject.toml`.

## Why now / what to export

The user-facing primitives in `torchlensmaker.__init__` form a clean
hierarchy that maps naturally onto trama:

- **Composites**: `Sequential`, `SubChain`, `Lens` (subclass of `Sequential`).
- **Leaf optical ops**: `RefractiveSurface`, `ReflectiveSurface`, `Aperture`,
  `ImagePlane`, `FocalPoint`.
- **Leaf source ops**: `Object`, `ObjectAtInfinity`, `PointSource`,
  `PointSourceAtInfinity`, `RaySource`, `GenericLightSource`.
- **Leaf kinematic ops**: `Gap`, `Translate`, `Translate2D`, `Translate3D`,
  `Rotate2D`, `RotateMixed`, `RotateX`, `RotateY`, `RotateZ`,
  `AbsolutePosition`, `AbsolutePosition2D`, `AbsolutePosition3D`.
- **Sub-leaf parameterized objects**: `SurfaceElement` subclasses
  (`SphereByRadius`, `SphereByCurvature`, `Conic`, `Asphere`, `Parabola`,
  `Plane`, `Disk`, `Square`, `XYPolynomial`, `PointSurface`, `Sphere`,
  `ImplicitDisk`), `MaterialModel` subclasses (`NonDispersiveMaterial`,
  `CauchyMaterial`, `SellmeierMaterial`, `LinearSegmentedMaterial`),
  `SamplerElement` subclasses, and `PositionGap` family.

Sub-modules such as surfaces, materials, and samplers are represented as child
nodes of their enclosing composite from the start, enabling correct parameter
sharing via the trama 0.2.0 parameter table.

## Design decisions

### D1 — Ports match `forward()` signatures

Each module's trama input and output ports are derived directly from its
`forward()` function arguments rather than from a unified wrapper type.
Different module categories therefore have different port structures:

- **Kinematic ops** (`Gap`, `Translate`, `Rotate*`, etc.): ports map to their
  kinematic transform arguments (e.g. `tf`).
- **Optical ops** (`RefractiveSurface`, `ReflectiveSurface`, etc.): ports map
  to their ray bundle and transform arguments (e.g. `rays`, `tf`).
- **Source ops** (`Object`, `PointSource`, etc.): ports match their own
  `forward()` signature, which differs from optical and kinematic ops.
- **Composites** (`Sequential`, `SubChain`, `Lens`): ports match the
  `forward()` signature of the contained module chain; wiring between children
  uses the actual argument names.

`SequentialData` is **not** used as a trama port type. The trama graph wires
the actual data fields (`rays`, `tf`, etc.) directly, keeping the exported
format self-describing and independent of internal wrapper conventions.

### D2 — Parameters always live in the top-level table

Trama's parameter model has one mode: every parameter lives in
`ExportedModel.parameters`, and every `Node.parameters` slot is a
`ParameterRef`. Map torchlensmaker types as follows when building a
`ParameterEntry`:

- `nn.Parameter` / `torch.Tensor` (any dimensionality, including 0-d scalars) →
  `TensorValue` with dtype preserved and `learnable=requires_grad`. Tensors are
  never coerced to bare Python scalars; dtype is always kept. Use
  `tensor.numpy().tobytes()` + `trama.encode_bytes` for raw bytes; `dtype`
  must come from `trama.tensor.KNOWN_DTYPES`.
- Bare Python scalars (`float`, `int`, `bool`) held directly on the module
  (not wrapped in `nn.Parameter`) → serialize as scalar values.
- Enum-like strings (`tir_mode`, material names like `"air"`, `"water"`) →
  plain Python `str`.
- Sub-modules that are *configuration* (e.g. samplers stored on a light
  source, or the inner `MaterialModel` of a `RefractiveSurface`) → child
  nodes in the trama tree (see D7).

The `to_node` pattern is uniform across every leaf:

```python
def to_node(self, *, node_id, ctx):
    x_ref, x_entry = pack_parameter(ctx, self.x, hint=f"{node_id}.x")
    return NodeFragment(
        root_id=node_id,
        nodes=[Node(
            id=node_id, parent=parent_of(node_id), op="Gap",
            inputs={"tf": ctx.input_refs["tf"]},
            outputs=["tf"],
            parameters={"x": x_ref},
            output_bindings=None,
        )],
        parameters=x_entry,
    )
```

When two modules share `self.x`, both calls to
`ctx.parameter_name(self.x, ...)` return the same name, both
fragments register the same entry under that name, and
`compose_fragment` merges them into one entry in the resulting
table. There is no inline-vs-shared decision to make at the call site.

### D3 — Sub-modules are deduplicated at export

Trama 0.2.0 supports identity sharing for parameters (via the top-level
parameter table) but not for sub-module instances. tlm models that share a
sub-module instance across multiple parents (e.g. one `SellmeierMaterial`
reused by six `RefractiveSurface`s) are deduplicated at export time: the
dedup pass makes a shallow copy of each reused module, producing structurally
independent sub-trees while preserving shared `nn.Parameter` identity. The
shared parameters then fall naturally into the top-level table via
`ctx.parameter_name` keying on `id(param)`.

The optimization-relevant invariants are preserved by construction:
the shared `nn.Parameter`s end up in the `parameters` table once
(because `id(param)` matches across the dedup'd copies), and every
copy of the sub-module references the same entries.
`optimizer.step()`, gradient flow, and `model.parameters()` behave
identically to the non-exported model.

The one user-visible change after round-trip: what was
`a_module is b_module` becomes `a_module == b_module` structurally
(same op, same parameter refs). Document this in the user guide; a
grep across `src/torchlensmaker` did not find any `is`-checks on
sub-modules.

Implementation: a `dedup_shared_modules(module) -> BaseModule` pass
in `src/torchlensmaker/export/dedup.py` walks the module tree, keys
sub-modules by `id()`, and makes shallow copies of any sub-module seen more
than once. The copy shares the original's `nn.Parameter` objects exactly
(same Python identity), so `ctx.parameter_name` keying on `id(param)` ensures
they map to a single entry in the parameter table.

### D4 — Registry keys mirror Python class names

Op names are simple `CamelCase` strings, identical to the Python class
name (e.g. `"Sequential"`, `"RefractiveSurface"`, `"SphereByRadius"`).
This keeps reading raw `.trama.json` files easy and matches what users
already expect from `repr()`.

### D5 — Single registry, populated at import-time

Add `torchlensmaker/export/registry.py` with a module-level
`tlm_registry: trama.OpRegistry` populated by registration calls scattered
across each user-facing module's file. Provide a public helper:

```python
def tlm_op_registry() -> trama.OpRegistry: ...
```

so users can call `trama.export_model(..., registry=tlm_op_registry())` and
`tlm.export_to_trama(model, path)` without knowing internal details.

### D6 — Mixin or protocol implementation?

`BaseModule` already inherits from `torch.nn.Module`. Add `to_node` /
`from_node` as **methods on `BaseModule`** with default implementations
that raise `NotImplementedError`, then override per class. This keeps the
trama API discoverable on every module via dot-access without changing
inheritance.

### D7 — Output bindings for composites

`Sequential.to_node` produces a composite node whose `output_bindings` wire
each declared output port to the corresponding output of the last child, using
the actual serialized field names derived from the `forward()` signature
(e.g. `rays`, `tf`):

```python
Node(
    id="root",
    op="Sequential",
    inputs={"rays": ValueRef(node="%input", output="rays"),
            "tf":   ValueRef(node="%input", output="tf")},
    outputs=["rays", "tf"],
    parameters={},
    output_bindings={"rays": ValueRef(node="root.<last_child>", output="rays"),
                     "tf":   ValueRef(node="root.<last_child>", output="tf")},
    parent=None,
)
```

`SubChain` and `Lens` follow the same pattern (with `op="SubChain"` /
`op="Lens"`). Optical elements that are trama composites for reconstruction
purposes (e.g. `RefractiveSurface`) use `output_bindings` that reference
`%input` directly, forwarding the actual port names through unchanged.

## Step-by-step implementation plan

The work is broken into sequenced PRs. Each step is independently testable.

### Step 0 — Scaffolding

1. Create `src/torchlensmaker/export/` with:
   - `__init__.py` — re-exports `tlm_op_registry`, `export_to_trama`,
     `import_from_trama`.
   - `registry.py` — defines the `tlm_registry` singleton and a
     `register_op(...)` helper.
   - `trama_helpers.py` — small utilities for tensor↔TensorValue
     conversion and dtype mapping. Includes a
     `pack_parameter(ctx, value, *, hint) -> tuple[ParameterRef,
     dict[str, ParameterEntry]]` helper that:
     1. calls `ctx.parameter_name(value, hint=hint)` to get a stable
        name,
     2. builds a `ParameterEntry` from `value` (handling tensors of any
        dimensionality with dtype preserved, bools, strings),
     3. returns the ref plus a one-entry dict the caller merges into
        its `NodeFragment.parameters`.
   - `dedup.py` — sub-module dedup pass: a function
     `dedup_shared_modules(module) -> BaseModule` that walks the tree,
     keys sub-modules by `id()`, and makes shallow copies of any
     sub-module seen more than once while preserving shared `nn.Parameter`
     identity.
2. Add `to_node` and `from_node` stubs to `BaseModule`
   (`src/torchlensmaker/core/base_module.py`) raising `NotImplementedError`.
3. Wire the public entry points in `src/torchlensmaker/__init__.py`:
   - `export_to_trama(model, path, *, indent=2)` — runs
     `dedup_shared_modules` first, then `trama.export_model(...)`,
     then `trama.dump`.
   - `import_from_trama(path) -> BaseModule`
   - `tlm_op_registry`
4. Add a single passing smoke test that imports `tlm.tlm_op_registry()`
   and asserts the returned object is an `OpRegistry` (initially empty).

### Step 1 — Composites: `Sequential`, `SubChain`, `Lens`

1. Implement `Sequential.to_node`: iterate over `self._modules`, recursively
   call `to_node` on each child with `node_id=f"{node_id}.{name}"` and
   `input_refs` wired to the previous child's outputs using the actual port
   names. The first child reads from `%input`. Wrap the child fragments with
   `compose_fragment` and emit the composite node with `output_bindings`.
2. Implement `Sequential.from_node`: receives `children` dict keyed by
   submodule name; rebuild via `Sequential(*children.values())`.
3. Implement `SubChain.to_node` / `from_node` as a thin wrapper around the
   inner `_sequential`.
4. Implement `Lens.to_node` / `from_node` (use `cemented`-style
   reconstruction is unnecessary; just feed the children in order).
5. Register all three ops in the registry with `is_composite=True`.
6. Tests: round-trip an empty `Sequential`, then nested
   `Sequential(SubChain(Sequential(...)))`. At this point all leaf children
   in the test must be trivial dummies (a stub op registered just for
   tests) because no real leaves are wired yet.

### Step 2 — Kinematic leaves

Lowest-risk leaves to wire next; they are pure parameters with no
sub-modules.

1. `Gap`: parameters `{x: TensorValue (learnable), reversed: bool}`. In
   `to_node`, run `self.x` through `pack_parameter(ctx, self.x,
   hint=f"{node_id}.x")` to get a `ParameterRef` and the entry to
   merge into the fragment's parameters table. `reversed` is a plain
   `bool`, also packed into the table (every parameter slot uses a
   ref; bools and strings are entries with scalar values).
2. `Translate`, `Translate2D`, `Translate3D`: parameters `{x, y, z?:
   TensorValue, reversed: bool}`. Each component goes through
   `pack_parameter` independently — crucial because users routinely share
   one component while leaving others independent.
3. `Rotate2D`, `RotateMixed`, `RotateX/Y/Z`: parameter `{angle: TensorValue
   (learnable), reversed: bool}`.
4. `AbsolutePosition*`: parameters `{x?, y?, z?: TensorValue}`.
5. Tests:
   - Round-trip a `Sequential(Gap(2.5, trainable=True), Translate(1, 2, 3))`,
     assert structural equality of the JSON and that
     `model.x.requires_grad == True` after import.
   - **Sharing test**: build `g = tlm.parameter(85.0, trainable=True);
     model = Sequential(Gap(g), Gap(g))`, round-trip, assert that
     `imported[0].x is imported[1].x` and that the JSON contains a
     non-empty top-level `parameters` table with one entry referenced
     twice.

### Step 3 — Surfaces and parameter-only sub-modules

Treat each `SurfaceElement` subclass as a leaf op, parameters captured
verbatim. Per surface:

1. `SphereByRadius`: `{diameter: TensorValue, R: TensorValue (learnable),
   anchors: TensorValue[2], scale: TensorValue}`.
2. `SphereByCurvature`, `Conic`, `Parabola`, `Plane`, `Disk`, `Square`:
   analogous, copy from each `__init__` signature.
3. `Asphere`, `XYPolynomial`: include polynomial coefficient tensors as
   `TensorValue`.
4. `PointSurface`, `Sphere` (implicit), `ImplicitDisk`: same treatment.
5. Add `PositionGap` family (`InnerGap`, `OuterGap`) — small leaf ops, just
   one `TensorValue` parameter.
6. Tests: import + export each surface in isolation, verify equality of
   stored tensor parameters.

### Step 4 — Materials

Each `MaterialModel` subclass becomes a leaf op:

- `NonDispersiveMaterial`: `{n: TensorValue (learnable)}`.
- `CauchyMaterial`: `{A, B, C, D: TensorValue}`.
- `SellmeierMaterial`: six `TensorValue`s.
- `LinearSegmentedMaterial`: two `TensorValue` arrays.

Special-case `material_from_indicio(name) -> MaterialModel`: round-trip by
storing the indicio key as a `string` parameter on a thin
`IndicioMaterial` wrapper op so the import side can call
`material_from_indicio(name)` to rehydrate.

### Step 5 — Optical leaves with sub-module parameters

These hold child modules (surface, materials). After the Step 0 dedup
pass, every `RefractiveSurface` instance owns a structurally independent
`MaterialModel` instance — but their `nn.Parameter`s may be shared via
the `parameters` table. The trama-side serialization is uniform: each
optical element is a composite node with named children, and trama's
parameter table handles parameter dedup.

1. `RefractiveSurface`: composite node with three children — `surface`,
   `material_in`, `material_out` — and parameters `{tir_mode: string}`.
   The composite declares its `forward()` ports (`rays`, `tf`), with
   `output_bindings` forwarding those ports from `%input` directly —
   children are reconstruction artifacts, not part of the dataflow.
   `from_node` reads `children["surface"]`, `children["material_in"]`,
   `children["material_out"]` and constructs a fresh
   `RefractiveSurface(surface, materials=(mat_in, mat_out),
   tir_mode=...)`.
2. `ReflectiveSurface`: composite with one child `surface` and no
   materials.
3. `Aperture`: composite with one child `surface` (a `Disk`) and no
   parameters.
4. `ImagePlane`: composite with child `surface=Disk(diameter)` and
   parameter `{magnification: TensorValue | None}`.
5. `FocalPoint`: leaf, no parameters, no children.

Trama allows composites whose `output_bindings` reference `%input`
directly — this is the correct encoding for nodes whose runtime behavior
is opaque to trama and whose children are reconstruction artifacts rather
than dataflow participants. Verify this passes `validate_structure` with a
focused unit test before wiring all optical elements.

### Step 6 — Samplers and light sources

Samplers are first-class child nodes (no string-key inlining). This is a
revision of the earlier draft's "inline samplers as a config string"
approach — making them proper child nodes is necessary because
`ExactSampler*` carries a tensor that may be shared.

1. Implement `SamplerElement` subclasses
   (`LinspaceSampler1D/2D`, `DiskSampler2D`, `ZeroSampler1D/2D`,
   `ExactSampler1D/2D`) as leaf ops; parameters are the sampler counts
   and (for `Exact*`) the sample tensors. Tensor parameters go through
   `ctx.share_parameter` so shared sample tensors dedup correctly.
2. Implement `GenericLightSource` and its concrete subclasses as
   composite ops with six sampler children plus the geometry parameters
   (`beam_angular_size`, `object_diameter`, `wavelength`, `material`,
   `reversed`, `source_idx`). Reuse the Step 5 composite pattern
   (`output_bindings` forwarding `%input` ports).
3. The `Object` / `ObjectAtInfinity` / `PointSource` /
   `PointSourceAtInfinity` / `RaySource` factories share most of their
   parameters; centralize the to_node helper in
   `light_sources/light_sources_elements.py`.

### Step 7 — Public API and ergonomics

1. Add `Sequential.to_trama(path)` and module-level
   `tlm.export_to_trama(model, path)` that internally call
   `trama.export_model(model, inputs=[...], outputs=[...],
   registry=tlm_op_registry())` followed by `trama.dump`, using the
   actual port names from the top-level module's `forward()` signature.
2. Add `tlm.import_from_trama(path)` which calls `trama.load` then
   `trama.import_model(loaded, registry=tlm_op_registry())`.
3. Update `src/torchlensmaker/__init__.py` to re-export the new symbols
   and add them to `__all__`.

### Step 8 — End-to-end validation

1. Add `tests/test_trama_roundtrip.py` covering each example in
   `examples/`:
   - `rainbow.py`
   - `cooke_triplet.ipynb`'s optical block
   - a `Sequential(ObjectAtInfinity, Lens, Gap, ImagePlane)` with all
     parameter shapes exercised (including a `LinearSegmentedMaterial` and
     an `Asphere`).
2. For each: build the model, call `export_to_trama → JSON →
   import_from_trama`, then compare:
   - `repr(model) == repr(reloaded)`
   - state-dict tensors element-wise close (`torch.allclose`)
   - one forward pass on a fixed input produces the same `RayBundle` outputs.
3. **Sharing-specific tests**:
   - One `SellmeierMaterial` with `trainable=True` shared across two
     lenses → after round-trip the six `nn.Parameter`s are still
     shared (`imported.L1.first_surface.material_in.B1 is
     imported.L2.first_surface.material_in.B1`), even though the
     `MaterialModel` *instances* are now distinct.
   - Run a 10-step optimization on the original and the round-tripped
     model with identical seeds; assert final parameter values are
     element-wise close.
   - Build a model with no shared anything; assert the JSON's
     top-level `parameters` table is empty.
4. Add a CLI smoke test that pipes the JSON output through `tramaschema |
   jsonschema validate` (or skip if the dependency is too heavy; rely on
   `trama.load`'s own validation).

### Step 9 — Documentation

1. Add `docs/src/trama_export.md` (or notebook) walking a user through
   exporting and reloading the Cooke triplet.
2. Add a short note to `README.md` near the existing format-export
   discussion.
3. Mention the format version (`trama.FORMAT_VERSION`) and the
   torchlensmaker registry version (introduce a `tlm_registry_version`
   constant in `export/registry.py`, `"0.1"`, bumped on breaking
   registry changes).

## Open questions / risks

1. **Validator rejection for composites with no internal dataflow**:
   Step 5 hinges on `validate_structure` accepting composites whose
   `output_bindings` reference `%input` directly. The validator code
   (`trama/validation_structure.py:_validate_output_bindings`) calls
   `_validate_ref` with `scope_parent=node.id`, which means
   `%input.<port>` inside the composite resolves to *this composite's*
   declared inputs. That should work. Confirm with a unit test before
   proceeding.
2. **Tensor parameters with `requires_grad`**: ensure the `learnable`
   flag is preserved across export/import and re-attached via
   `nn.Parameter` on the import side. With shared parameters, the
   table entry's `learnable` is canonical — the `parameters` table
   pre-build must wrap `nn.Parameter` *before* any node references it
   so that all references see the same `nn.Parameter` object.
3. **Sub-module dedup correctness**: the export-time
   `dedup_shared_modules` pass must handle deeply nested sharing
   (e.g. a shared `MaterialModel` whose internal kernels are also
   shared with another module's kernels). Recursive `id()`-keyed
   walking is correct but needs a focused test that constructs such
   a case.
4. **`is`-checks on sub-modules in tlm**: the round-trip turns
   `a is b` into `a == b` for shared sub-modules. A grep across
   `src/torchlensmaker` did not find any `is`-comparisons on
   sub-modules, but new code should not introduce any. Consider
   adding a lint rule or doc note.
5. **Backward compatibility**: trama format version is `"0.2"` after
   the shared-parameters feature. Bump the torchlensmaker registry
   version on every additive op change; refuse loads with a higher
   version than what's installed. Add a `format_compat_check` in
   `import_from_trama`.

## Out of scope (for the first iteration)

- ONNX or `torch.export` integration.
- Optimizer state, training history, or `OptimizationRecord`
  serialization.
- Custom user-defined `BaseModule` subclasses outside the public
  `tlm.*` API. (Document the registry extension API in Step 9 as a
  follow-up.)
- Streaming / chunked loading of huge models.
- Sub-module identity preservation across round-trip (intentionally
  unsupported; see D3).
