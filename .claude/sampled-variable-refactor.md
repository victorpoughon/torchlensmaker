# Plan: `SampledVariable` abstraction + `RayBundle` / kernel refactor

## Context

`RayBundle` (torchlensmaker/src/torchlensmaker/core/ray_bundle.py) currently flattens four sampling variables (pupil, field, wavel, source) into 8 separate fields: `pupil`, `pupil_idx`, `field`, `field_idx`, `wavel`, `wavel_idx`, `source_idx`, plus `P`/`V`. Adding the per-axis "domain + samples" data needed for `split_by` (and for labeling empty cells in spot-diagram grids) would push this to 16+ fields. That's noisy, error-prone, and doesn't generalize to any future sampling variable.

The proposed abstraction: a `SampledVariable` dataclass holding the four tensors that together describe one variable, and a `RayBundle` that holds four of them (one per variable). The geometry kernels in `source_geometry_kernels.py` already compute everything needed internally — they just don't return the per-axis pieces — so refactoring them to return `SampledVariable`s closes the loop and removes the scatter/arange adapter currently sitting between kernel and bundle.

This plan is the prerequisite for `RayBundle.split_by`. After this lands, `split_by` becomes a small additive method that iterates over each variable's `domain_idx`. The end goal is fully configurable spot-diagram grids that can label empty cells (where every ray was filtered out).

## Locked design decisions

1. **Class name**: `SampledVariable`.
2. **`source` values**: uniform with the other variables — `source.values = source.idx.to(float_dtype)`. Slight redundancy, but no `Optional`, no special branches in mask/cat/render code.
3. **`cat` semantics**: single rule for all four variables — *union the domains, assert `torch.allclose` on overlap*. Disjoint domains union; same-source overlaps satisfy the assertion trivially. No per-instance flag.
4. **ONNX rename**: kernel flat output names will change (`angular_coordinates` → `angular.values`, etc.). Internal-only, accepted.
5. **Step order**: `SampledVariable` → kernel I/O type → kernel + element refactor → `RayBundle` refactor. The bundle change comes last so it's a pure mechanical rewire of fully-formed kernel/element outputs, with no throwaway scatter/arange logic in any intermediate state.
6. **Public API**: `SampledVariable` is exported from `torchlensmaker/__init__.py` so users can write `tlm.SampledVariable`.
7. **Method name**: the bundle's partition method is `split_by` (not `group_by`).

### One remaining minor choice (default unless you object)

**Wavel shape.** Today `wavel` is `(N,)` while pupil/field are `(N,)` in 2D and `(N, 2)` in 3D. Plan keeps heterogeneous trailing dims — each `SampledVariable` carries its own `values.shape[1:]`, no unification to a contrived `D=1` for wavel. Yell if you'd prefer the uniform `(N, D)` everywhere.

## Architecture map (before reading the steps)

The data flows through three layers between the meshgrid math and the `RayBundle`:

```
source_geometry_kernels.py          ← physical concepts: angular / spatial / wavelength
  ↓
source_geometry_elements.py         ← optical concepts: pupil / field / wavel
                                       (does the angular↔pupil rename, the
                                        rad→deg conversion, and — for objects
                                        at infinity — the angular↔spatial swap)
  ↓
light_sources_elements.py           ← attaches source_idx and constructs the RayBundle
  ↓
RayBundle
```

Two **kernels** in `source_geometry_kernels.py`:
- `ObjectGeometry2DKernel`, `ObjectGeometry3DKernel`.

Four **geometry elements** in `source_geometry_elements.py`, all reusing the two kernels above:
- `ObjectGeometry2D` / `ObjectGeometry3D` — near-field: `pupil = angular`, `field = spatial`.
- `ObjectAtInfinityGeometry2D` / `ObjectAtInfinityGeometry3D` — at infinity: `pupil = spatial`, `field = angular` (**swapped**).
- All four apply `torch.rad2deg(angular)` so the angular variable surfaces in degrees.

The plan treats kernel and element layers explicitly because they speak different vocabularies.

## Step 1 — Create `SampledVariable`

**Scope**: pure addition, no consumers yet.

New file `torchlensmaker/src/torchlensmaker/core/sampled_variable.py`:

```python
@dataclass
class SampledVariable:
    values: BatchNDTensor          # (N, ...) per-ray physical values
    idx: IndexTensor               # (N,) per-ray indices into the domain
    domain_values: BatchNDTensor   # (M, ...) per-axis physical values
    domain_idx: IndexTensor        # (M,) sorted unique int64

    @classmethod
    def create(cls, values, idx, domain_values, domain_idx) -> Self: ...
        # dtype/device/shape assertions:
        # - values.dtype == domain_values.dtype (float)
        # - idx.dtype == domain_idx.dtype == int64
        # - all on same device
        # - values.shape[1:] == domain_values.shape[1:]
        # - idx.shape == values.shape[:1]
        # - domain_idx.shape == domain_values.shape[:1]
        # - domain_idx is sorted unique

    @classmethod
    def empty(cls, value_shape: tuple[int, ...], dtype, device) -> Self:
        # values: (0, *value_shape), idx: (0,), domain_values: (0, *value_shape),
        # domain_idx: (0,). value_shape == () for scalar variables (wavel, source).

    def mask(self, valid: MaskTensor) -> Self:
        # values/idx filtered; domain_values/domain_idx unchanged
        return SampledVariable(
            values=self.values[valid],
            idx=self.idx[valid],
            domain_values=self.domain_values,
            domain_idx=self.domain_idx,
        )

    def cat(self, other: Self) -> Self:
        # Concat values and idx straight.
        # Build merged domain: torch.unique(torch.cat([self.domain_idx, other.domain_idx]))
        # For each domain key, look up domain_values from whichever side has it;
        # if both sides have it, assert torch.allclose and take either.
        # Empty-input shortcuts: cat(empty, full) → full, cat(full, empty) → full.
```

### Public API

`torchlensmaker/src/torchlensmaker/__init__.py` adds `from torchlensmaker.core.sampled_variable import SampledVariable` and includes it in `__all__` (or its existing equivalent).

### Tests (`core/tests/test_sampled_variable.py`)

- `create` with mismatched dtypes/devices/shapes → `AssertionError`.
- `create` with non-sorted or non-unique `domain_idx` → `AssertionError`.
- `mask` preserves domain (idx and values).
- `cat` overlapping domain matching → unchanged domain.
- `cat` overlapping domain mismatched → raises.
- `cat` disjoint domain (e.g. {0} ∪ {5}) → sorted union {0, 5}.
- `cat` with empty left or right → returns the non-empty side intact.
- `tlm.SampledVariable` import works (smoke).

### Verification

`uv run pytest src/torchlensmaker/core/tests/test_sampled_variable.py`

## Step 2 — Make `SampledVariable` a kernel I/O type

**Scope**: infrastructure only. No kernel logic changes yet.

- `core/functional_kernel.py`:
  - `KernelIOType: TypeAlias = torch.Tensor | Tf | SampledVariable`.
  - `torch.export.register_dataclass(SampledVariable)` next to the existing `Tf` registration (line 30).
  - Verify `kernel_flat_io` (line 37) and `kernel_flat_names` (line 52) handle the new dataclass — they already use `is_dataclass`/`astuple`/`fields` generically, so this should be no-op in code, but cover with a test.
- New test (`core/tests/test_functional_kernel.py` — create the file): a tiny dummy kernel that returns a `SampledVariable`. Verify (a) `kernel_flat_names` produces `var.values`, `var.idx`, `var.domain_values`, `var.domain_idx`, and (b) ONNX export round-trip works and the loaded model has those same flat output names.

### `dynamic_shapes` interaction (worth noting, no code change in this step)

- `dynamo_dynamic_shapes` (line 108) consumes only `flat_input_names + flat_param_names` — never outputs. Output dynamic dims are inferred from the trace, so adding a dataclass *output* type needs no changes here.
- Kernels with a dataclass *input* aren't introduced by this refactor (geometry kernel inputs stay raw tensor `*_samples`). If a future kernel needs that, hierarchical keys (`{"my_var": {"values": {0: "N"}}}`) and a recursive expansion in `dynamo_dynamic_shapes` would be the extension — out of scope here.

### Verification

- `uv run pytest src/torchlensmaker/core/tests/`
- Spot-check one existing ONNX export still works (`uv run pytest src/torchlensmaker/light_sources/tests/test_source_type_kernels.py`).

## Step 3 — Geometry kernels and elements return `SampledVariable`

**Scope**: kernel + element API change. The `RayBundle` still has flat fields, so `light_sources_elements.py` becomes a temporary *unpacker* that pulls `.values` / `.idx` out of each element-returned `SampledVariable` to feed the old `RayBundle.create` signature. Domain info is computed and intentionally dropped here — step 4 wires it through.

This step touches three files and is internally split into three sub-changes that should land in a single commit:

### Step 3a — `source_geometry_kernels.py` (kernel layer, angular/spatial vocabulary)

Both `ObjectGeometry2DKernel` and `ObjectGeometry3DKernel`:

- `outputs` dict changes from 8 entries to 5: `P`, `V`, `angular: SampledVariable`, `spatial: SampledVariable`, `wavel: SampledVariable`. Note: kernel-layer naming stays angular/spatial — the pupil/field rename is the element layer's job.
- `apply()` constructs the three `SampledVariable`s before returning. The pieces are already locals:
  - `angular`: `domain_values = angular_coords` (pre-meshgrid); `domain_idx = angular_idx = arange(Na)`; `values = angular_coords_full` (post-meshgrid); `idx = angular_idx_full`. Same pattern for `spatial` and `wavel`.
  - **No scatter is needed** — every piece is already a local.
- `dynamic_shapes` declarations on inputs (`{"angular_samples": {0: "N_angular"}, ...}`) stay exactly as they are — inputs are unchanged.
- 2D-kernel: type-correctness opportunity. The 2D kernel's old `outputs` declared `angular_coordinates: Batch2DTensor` (= shape `... 2`) but actual shape is `(N,)`. With the rewrite to `SampledVariable`, this becomes `angular: SampledVariable` and the inconsistency disappears. (Already noted as fixed by user.)

### Step 3b — `source_geometry_elements.py` (element layer, pupil/field vocabulary)

All four classes (`ObjectGeometry2D`, `ObjectGeometry3D`, `ObjectAtInfinityGeometry2D`, `ObjectAtInfinityGeometry3D`):

- `forward()` return type changes from an 8-tuple of raw tensors to a 5-tuple: `(P, V, pupil_sv, field_sv, wavel_sv)`.
- `forward()` body: receive `angular_sv, spatial_sv, wavel_sv` from the kernel and reshape them into pupil/field naming + apply the rad→deg conversion to whichever variable is the angular one. The conversion has to be applied to **both `values` and `domain_values`** (idx and domain_idx are unitless and unchanged):

  ```python
  def rad2deg_var(var: SampledVariable) -> SampledVariable:
      return SampledVariable(
          values=torch.rad2deg(var.values),
          idx=var.idx,
          domain_values=torch.rad2deg(var.domain_values),
          domain_idx=var.domain_idx,
      )
  ```

  - **Near-field** (`ObjectGeometry2D`/`3D`): `pupil_sv = rad2deg_var(angular_sv)`, `field_sv = spatial_sv`.
  - **At infinity** (`ObjectAtInfinityGeometry2D`/`3D`): `pupil_sv = spatial_sv`, `field_sv = rad2deg_var(angular_sv)`. (Note the swap is unchanged from today; the angular tensor still carries field-sample indices because the kernel call swapped its `angular_samples`/`spatial_samples` inputs.)

  Whether the helper lives module-level or inline is implementation choice; either is fine.

### Step 3c — `light_sources_elements.py` (intermediate unpacker)

`GenericLightSource.forward` becomes:

```python
P, V, pupil_sv, field_sv, wavel_sv = geometry(tf, pupil_samples, field_samples, wavel_samples)
if self.reversed:
    V = -V
N = P.shape[0]
source_idx = torch.full((N,), self.source_idx, dtype=torch.int64, device=device)
return RayBundle.create(
    P=P, V=V,
    pupil=pupil_sv.values, field=field_sv.values, wavel=wavel_sv.values,
    pupil_idx=pupil_sv.idx, field_idx=field_sv.idx, wavel_idx=wavel_sv.idx,
    source_idx=source_idx,
)
```

`pupil_sv.domain_values` / `pupil_sv.domain_idx` (etc.) are computed by the kernel/element but discarded in this step. Step 4 wires them through.

### Tests

- Update kernel-level tests (`src/torchlensmaker/light_sources/tests/test_source_type_kernels.py`) to consume the new 5-output schema.
- Update or add element-level tests if any exist; otherwise no new test file is mandatory at this step (the integration coverage in `tests/test_elements.py` exercises the full chain).

### Verification

- `uv run pytest src/ tests/` — full regression must pass.
- Re-run ONNX export tests; spot-check one exported model file's output names show `angular.values`, `angular.idx`, `angular.domain_values`, `angular.domain_idx`, `spatial.*`, `wavel.*`.

## Step 4 — Refactor `RayBundle` to hold `SampledVariable`s

**Scope**: mechanical rewire of the bundle and its consumers — no new logic. The unpacker added in step 3c collapses to direct passthrough.

### `RayBundle` schema (ray_bundle.py)

```python
@dataclass
class RayBundle:
    P: BatchNDTensor
    V: BatchNDTensor
    pupil: SampledVariable
    field: SampledVariable
    wavel: SampledVariable
    source: SampledVariable
```

- `create`: takes 4 `SampledVariable`s. Asserts `P.dtype == V.dtype == pupil.values.dtype == ...` and same device across everything.
- `empty(dim, dtype, device)`: builds 4 empty `SampledVariable`s with the right `value_shape` per variable:
  - pupil: `()` for 2D, `(2,)` for 3D.
  - field: `()` for 2D, `(2,)` for 3D.
  - wavel: `()`.
  - source: `()`.
- `mask(valid)`: delegates to each `SampledVariable.mask(valid)`. P and V slice as today.
- `cat(other)`: delegates to each `SampledVariable.cat(other.<var>)`. P and V cat as today.
- `replace`: unchanged (still works via `dataclasses.replace`).

### Consumer updates (read sites)

Confirmed by survey:

- `viewer/render_model_trace.py:42-51`:
  - `rays.pupil` → `rays.pupil.values`
  - `rays.field` → `rays.field.values`
  - `rays.wavel` → `rays.wavel.values`
  - `rays.source_idx` → `rays.source.idx`
- `optical_surfaces/refractive_surface.py:74-75`:
  - `rays.wavel` (×2) → `rays.wavel.values`
- `light_targets/image_plane.py:98`:
  - `rays_propagated.field` → `rays_propagated.field.values`
- `tests/test_elements.py:33-49, 318-319`:
  - Each `rays.<var>` → `rays.<var>.values`
  - Each `rays.<var>_idx` → `rays.<var>.idx`

### Construction site

`light_sources_elements.py` `GenericLightSource.forward` simplifies to:

```python
P, V, pupil, field, wavel = geometry(tf, pupil_samples, field_samples, wavel_samples)
if self.reversed:
    V = -V
N = P.shape[0]
source = SampledVariable(
    values=torch.full((N,), float(self.source_idx), dtype=dtype, device=device),
    idx=torch.full((N,), self.source_idx, dtype=torch.int64, device=device),
    domain_values=torch.tensor([float(self.source_idx)], dtype=dtype, device=device),
    domain_idx=torch.tensor([self.source_idx], dtype=torch.int64, device=device),
)
return RayBundle.create(P=P, V=V, pupil=pupil, field=field, wavel=wavel, source=source)
```

The unpacker from step 3c is gone — element outputs flow straight through.

### Tests

Replace `test_ray_bundle.py`. Cover:

- Round-trip `mask(all_true)` preserves all four sub-objects (values, idx, domain_values, domain_idx).
- `cat` with disjoint sources unions correctly.
- `cat` with conflicting wavel `domain_values` raises.
- After `mask` that drops all rays of one `field_idx` value, that value still appears in `field.domain_idx` and `field.domain_values`.
- Invariant for a freshly-created bundle: `pupil.values[i] == pupil.domain_values[searchsorted(pupil.domain_idx, pupil.idx[i])]` (and likewise for field/wavel/source).
- Empty bundle has empty `SampledVariable`s with correct shapes per dim.

### Verification

- `uv run pytest src/torchlensmaker/core/tests/test_ray_bundle.py`
- `uv run pytest src/ tests/` — full regression. Render, image_plane, refractive_surface consumers must stay green.

## Files modified (cumulative)

- **Step 1 (new)**: `core/sampled_variable.py`, `core/tests/test_sampled_variable.py`, `torchlensmaker/__init__.py` (add export).
- **Step 2**: `core/functional_kernel.py`, `core/tests/test_functional_kernel.py` (new).
- **Step 3**: `light_sources/source_geometry_kernels.py`, `light_sources/source_geometry_elements.py`, `light_sources/light_sources_elements.py` (intermediate unpacker), `light_sources/tests/test_source_type_kernels.py` (new output schema).
- **Step 4**: `core/ray_bundle.py`, `viewer/render_model_trace.py`, `optical_surfaces/refractive_surface.py`, `light_targets/image_plane.py`, `light_sources/light_sources_elements.py` (final simplification), `core/tests/test_ray_bundle.py`, `tests/test_elements.py`.

## Commit strategy

One commit per step, each independently green (`uv run pytest src/ tests/` passes after each):

1. `feat(core): add SampledVariable dataclass`
2. `feat(kernel): allow SampledVariable as kernel I/O type`
3. `refactor(geometry): kernels and elements return SampledVariable`
4. `refactor(ray_bundle): hold SampledVariables, update consumers`

## After step 4 — what's unlocked, and the spot-diagram caveat

`RayBundle.split_by(var1, var2=None) -> list[list[RayBundle]]` becomes a small additive method that iterates over each variable's `SampledVariable.domain_idx`. Empty cells (where every ray was filtered) appear as empty `RayBundle`s with their `domain_values`/`domain_idx` intact — the spot-diagram grid can label them by reading `cell.field.domain_values[k]` etc.

**Caveat for the spot-diagram refactor that comes after this:** `output.rays_image` (image-plane coordinates used by `analysis/spot_diagram.py`) is a **separate** parallel tensor carried alongside `RayBundle`, not inside it. `split_by` partitions a `RayBundle`, not its parallel sidecars. The follow-up spot-diagram refactor will need to either:
- Apply the same masks `split_by` derives to `rays_image`, **or**
- Lift `rays_image` (and any other parallel sidecar) into `RayBundle` itself.

That's a separate decision and out of scope for this plan.
