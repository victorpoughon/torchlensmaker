# OpticalTrace Refactor Plan

This document captures the design and implementation plan for the new
`OpticalTrace` data structure and the surrounding refactor that unifies
forward and trace evaluation modes, introduces non-shrinking ray bundles,
and adds first-class support for differentiable optical path length (OPL).

## Goals

1. Differentiable optical path length, including over arbitrary
   sub-sequences of a model, usable as a term in a loss function.
2. First-class accumulating recording of a forward pass (path, kinematics,
   per-element records) replacing the current dual forward / trace mode.
3. Keep the door open for future path-aware features (wavefront, OPD,
   Fermat-style optimization) without redesign.
4. Allow non-linear topology (branching, e.g. beam splitters) without
   forcing complexity on the common linear case.

## Locked decisions

- A new `OpticalTrace` data structure replaces both `SequentialData` and
  `ModelTrace`. Element-level `trace()` methods, `trace_model()`, and the
  forward-hook mechanism are removed.
- `OpticalTrace` is a flat `OrderedDict[str, OpticalTraceNode]` plus
  `bundles: dict[str, RayBundle]` and `tfs: dict[str, Tf]` storage
  tables. Each node carries an `ElementRecord`, parent pointers, and
  `bundle_id` / `tf_id` strings that index into the storage tables.
  Inheritance is expressed by sharing IDs across nodes — there is no
  `None`-as-inherit convention in the data layer.
- Linear-chain composition stays. No new container modules
  (`Branch`/`Parallel`/`Merge`). `Sequential` and `SubChain` only.
- Multi-parent nodes are allowed in two forms:
  - **Concatenating merge**: different row layouts stacked into a longer
    bundle. Used today by multi-source models via `RayBundle.cat`.
    Supported.
  - **Identifying merge**: two bundles with the same source-aligned row
    layout combined such that row `k` represents the same physical ray
    that took two paths (e.g. interferometric recombination). Requires
    reconciling per-ray identity across paths. **Out of scope.**
- Branching that does not recombine (e.g. a beam splitter feeding two
  independent downstream chains) is allowed.
- Optical path and kinematic chain are **derived views** over the trace,
  not stored fields. A `linear_path()` utility raises if the trace is not
  linear; a more general `paths()` returns a list.

## Data structures

### `ElementRecord` family

```python
class ElementRecord:
    kind: str  # discriminator; each subclass narrows to a Literal

@dataclass
class SourceRecord(ElementRecord):
    kind: Literal["source"] = "source"

@dataclass
class CollisionRecord(ElementRecord):
    t: BatchTensor
    normals: BatchNDTensor
    valid: MaskTensor
    points_local: BatchNDTensor
    points_global: BatchNDTensor
    rsm: BatchTensor
    tf_surface: Tf
    kind: Literal["collision"] = "collision"

@dataclass
class ApertureRecord(ElementRecord):
    valid_update: MaskTensor
    kind: Literal["aperture"] = "aperture"

@dataclass
class FocalRecord(ElementRecord):
    loss: ScalarTensor
    target_tf: Tf
    kind: Literal["focal"] = "focal"

# Plus: ImagePlaneRecord (kind="image_plane"), GapRecord
# (kind="gap", kinematic spacer), etc., as elements are migrated.
```

Records replace the parallel dicts on `ModelTrace`. They are the only
polymorphic per-element data; consumers dispatch on `record.kind`
(or on `isinstance` when richer pattern matching is needed). The
`kind` field is placed last on each subclass so existing positional
fields keep their definitionless-default property.

### `OpticalTraceNode`

```python
@dataclass
class OpticalTraceNode:
    key: str                      # this node's own key (== dict key in trace.nodes)
    record: ElementRecord
    parents: set[str]             # set() for sources, {prev} for sequential
    bundle_id: str                # key into trace.bundles
    tf_id: str                    # key into trace.tfs
```

- A node represents an *event* (an element fired). It always names a
  resolved bundle and tf via string IDs into the trace-level storage
  tables.
- Inheritance is expressed by sharing IDs: if this node didn't change
  the bundle, its `bundle_id` equals one of its parents' `bundle_id`.
  Same for `tf_id`.
- Invariant: `node.bundle_id == node.key` iff this node introduced a
  new bundle (likewise for `tf_id`). So "did element K change the
  bundle?" is one string comparison.
- `parents` is a `set[str]` so duplicate parents are impossible by
  construction. Order is not meaningful; row-layout ordering of a
  concatenating merge lives in the bundle itself.

### `OpticalTrace`

```python
@dataclass
class OpticalTrace:
    dim: int
    dtype: torch.dtype
    device: torch.device

    nodes: OrderedDict[str, OpticalTraceNode]
    bundles: dict[str, RayBundle]
    tfs: dict[str, Tf]

    # Resolution
    def bundle_at(self, key: str) -> RayBundle:
        return self.bundles[self.nodes[key].bundle_id]
    def tf_at(self, key: str) -> Tf:
        return self.tfs[self.nodes[key].tf_id]

    # Build
    def append(
        self,
        key: str,
        record: ElementRecord,
        parents: set[str],
        new_bundle: RayBundle | None,   # None = inherit parent's bundle_id
        new_tf: Tf | None,              # None = inherit parent's tf_id
    ) -> None: ...
        # If new_bundle is not None: store it as bundles[key] and set
        # node.bundle_id = key.
        # If new_bundle is None: copy bundle_id from the (single) parent.
        # When parents has more than one element, new_bundle MUST NOT be
        # None — multi-parent inheritance is ambiguous. Same rules for tf.

    # Queries
    def keys(self) -> list[str]: ...
    def is_linear(self) -> bool: ...
```

Trace-level invariants (cheap to assert in `__post_init__` or in
tests):

- Every `node.bundle_id` is in `bundles`; every `node.tf_id` is in `tfs`.
- `node.bundle_id == node.key` iff this node introduced a new bundle
  (same for `tf_id`). No orphan entries in `bundles` / `tfs`.
- Node keys are unique across the trace.
- Every key in `node.parents` exists in `nodes`.

### Derived chain views

```python
@dataclass
class OpticalPath:
    """A linear walk through the trace. All bundles share row layout N."""
    nodes: list[OpticalTraceNode]      # in order
    keys: list[str]

    def segment_lengths(self) -> BatchTensor: ...      # (K-1, N)
    def segment_n(self) -> BatchTensor: ...            # (K-1, N)
    def segment_valid(self) -> MaskTensor: ...         # (K-1, N)
    def opl(self) -> BatchTensor: ...                  # (N,) per-ray

@dataclass
class KinematicChain:
    """A linear walk through the trace, frames only."""
    tfs: list[Tf]
    keys: list[str]


def linear_path(trace: OpticalTrace, start: str, end: str) -> OpticalPath:
    """Walk parent pointers from end back to start. Raise if non-linear."""

def paths(trace: OpticalTrace, start: str, end: str) -> list[OpticalPath]:
    """All linear paths from start to end. For trees/DAGs."""

def linear_kinematic_chain(trace: OpticalTrace, start: str, end: str) -> KinematicChain: ...
```

## Element protocol

Two-tier shape, kept narrow:

- **Kernel `forward`** (local, narrow, testable): returns *only what the
  element actually produces*. Different element kinds have different
  signatures — there is no shared "always returns bundle and tf" rule.
- **`extend_optical_trace(trace, key, parents) -> trace`** (chain API):
  thin per-element method that reads its input from
  `trace.bundle_at(...)` / `trace.tf_at(...)` for each parent, calls
  its kernel, and appends a node with `new_bundle` and `new_tf` set to
  exactly what the element type produces. The caller (`Sequential`) is
  responsible for supplying `key` (from `mod.trace_key or module_key`)
  and the correct `parents` set. No identity comparison, no implicit
  dedup — each element knows its own semantics.

Per-element shapes:

```python
# Refractive / reflective surface — produces both a new bundle and a new tf
class OpticalSurfaceElement(BaseModule):
    def forward(self, rays: RayBundle, tf: Tf) -> tuple[RayBundle, CollisionRecord]:
        ...

    def extend_optical_trace(
        self, trace: OpticalTrace, key: str, parents: set[str]
    ) -> OpticalTrace:
        (parent_key,) = parents  # sequential: exactly one parent
        rays_in = trace.bundle_at(parent_key)
        tf_in = trace.tf_at(parent_key)
        new_rays, record = self.forward(rays_in, tf_in)
        trace.append(
            key=key,
            record=record,
            parents=parents,
            new_bundle=new_rays,
            new_tf=record.tf_next,
        )
        return trace


# Aperture — produces a new bundle (updated valid) but no new tf
class Aperture(BaseModule):
    def forward(self, rays: RayBundle, tf: Tf) -> tuple[RayBundle, ApertureRecord]:
        ...

    def extend_optical_trace(
        self, trace: OpticalTrace, key: str, parents: set[str]
    ) -> OpticalTrace:
        (parent_key,) = parents
        rays_in = trace.bundle_at(parent_key)
        tf_in = trace.tf_at(parent_key)
        new_rays, record = self.forward(rays_in, tf_in)
        trace.append(
            key=key,
            record=record,
            parents=parents,
            new_bundle=new_rays,
            new_tf=None,
        )
        return trace


# Kinematic spacer — produces a new tf but no new bundle
class Gap(BaseModule):
    def forward(self, tf: Tf) -> tuple[Tf, GapRecord]:
        ...

    def extend_optical_trace(
        self, trace: OpticalTrace, key: str, parents: set[str]
    ) -> OpticalTrace:
        (parent_key,) = parents
        tf_in = trace.tf_at(parent_key)
        new_tf, record = self.forward(tf_in)
        trace.append(
            key=key,
            record=record,
            parents=parents,
            new_bundle=None,
            new_tf=new_tf,
        )
        return trace


# Light target (e.g. focal point) — produces neither a new bundle nor a new tf,
# only a record carrying the loss
class FocalPoint(BaseModule):
    def forward(self, rays: RayBundle, tf: Tf) -> FocalRecord:
        ...

    def extend_optical_trace(
        self, trace: OpticalTrace, key: str, parents: set[str]
    ) -> OpticalTrace:
        (parent_key,) = parents
        rays_in = trace.bundle_at(parent_key)
        tf_in = trace.tf_at(parent_key)
        record = self.forward(rays_in, tf_in)
        trace.append(
            key=key,
            record=record,
            parents=parents,
            new_bundle=None,
            new_tf=None,
        )
        return trace


# Light source — produces a fresh bundle, no parents
class LightSourceBase(BaseModule):
    def forward(self, tf: Tf) -> tuple[RayBundle, SourceRecord]:
        ...

    def extend_optical_trace(
        self, trace: OpticalTrace, key: str, parents: set[str]
    ) -> OpticalTrace:
        # Source nodes are called with parents=set(). The source's tf is
        # its placement frame (a per-source attribute / external input).
        # The new bundle is fresh; bundle_id and tf_id both equal the
        # source's own key. When more than one source exists in a model,
        # downstream elements consume a concatenating merge of all live
        # sources (see multi-source notes below).
        ...
```

The kernel signatures are deliberately heterogeneous. A surface kernel
returns `(RayBundle, CollisionRecord)`. An aperture kernel returns
`(RayBundle, ApertureRecord)`. A spacer kernel returns
`(Tf, GapRecord)`. A target kernel returns just a record. This is
honest about what each kind of element actually does.

## Migration plan

Each step is independently shippable. After each step the codebase
should pass tests.

### Step 1 — Add `valid` and `n` to `RayBundle`

DONE.

### Step 2 — Refactor `ModelTrace` to use `ElementRecord`

Pure refactor of the existing trace mode: parallel dicts collapse into
a single `OrderedDict[str, ElementRecord]`. No graph structure yet,
no changes to forward semantics, no new files outside the records
module. The hook mechanism and per-element `trace()` methods stay.

Files:
- `core/element_records.py` (new)
- `sequential/model_trace.py` (modified)

Guiding rule for record contents (so step 3 doesn't churn them):

> A record carries only the data that is *intrinsic to the event the
> element produced*. It does not carry upstream context.

In particular:
- `CollisionRecord` carries `t`, `normals`, `valid`, `points_local`,
  `points_global`, `rsm`, `tf_surface`, the produced bundle, the
  produced `tf_next`, and a reference to the surface element. It does
  **not** carry `input_rays` or `input_tf` — those live on the parent
  node and are reached via lookup helpers.
- The produced bundle and `tf_next` sit on the record in step 2 and
  migrate to `trace.bundles` / `trace.tfs` in step 3 with no field
  rename — same data, new home.

Substeps:
1. Define `ElementRecord` base class with the `kind` discriminator,
   plus the concrete subclasses: `SourceRecord`, `CollisionRecord`,
   `ApertureRecord`, `FocalRecord`, `ImagePlaneRecord`, etc. (Add
   `GapRecord` only when an actual `Gap` element is migrated.)
2. Replace `ModelTrace`'s parallel dicts (`input_rays`, `output_rays`,
   `collisions`, `input_joints`, `output_joints`, `surfaces`,
   `focal_points`) with a single
   `nodes: OrderedDict[str, ElementRecord]`. Keep `dim`.
3. Add lookup helpers on `ModelTrace`: `bundle_in_at(k)`,
   `bundle_out_at(k)`, `tf_in_at(k)`, `tf_out_at(k)`. They walk
   `nodes` in insertion order to find the upstream context. These
   helpers are the only API consumers should use; they become
   parent-aware in step 3 with no signature change.
4. Update each element's existing `trace()` method to construct one
   record and append it via a single `add_node(key, record)` call.
   Drop the per-field `add_*` methods.
5. Migrate the viewer and analysis modules onto records via the
   lookup helpers. No consumer should `isinstance`-dispatch records
   in step 2 if a helper covers the access pattern; reserve `kind`
   dispatch for genuinely polymorphic readers (e.g. the viewer's
   per-element render switch).

Tests:
- Existing forward / trace integration tests pass unchanged.
- New tests: round-trip of each record kind, `kind`-discriminator
  invariants, lookup helpers behave correctly across the existing
  fixtures.

### Step 3 — Introduce `OpticalTrace` / `OpticalTraceNode` and derived views

Additive step: define the graph structure and OPL machinery, but do
not wire into `Sequential`. Records from step 2 are reused; the only
change to records is that the produced bundle and `tf_next` move off
the record into the trace-level storage tables.

Files added:
- `core/optical_trace.py` (or `sequential/optical_trace.py`)
- `core/optical_path.py`

Substeps:
1. Move the produced bundle and `tf_next` fields off the records into
   the trace's `bundles` / `tfs` storage tables. Records keep
   everything else.
2. Define `OpticalTraceNode` (`key`, `record`, `parents`, `bundle_id`,
   `tf_id`) and `OpticalTrace` (`dim`, `dtype`, `device`, `nodes`,
   `bundles`, `tfs`) with `append`, `bundle_at`, `tf_at`, `is_linear`.
   Enforce the trace-level invariants in `__post_init__` or a
   `validate()` method.
3. Define `OpticalPath` and `KinematicChain` derived views, plus
   `linear_path()`, `linear_kinematic_chain()`, and `paths()`.
4. Implement `OpticalPath.opl()` and `OpticalPath.segment_*` methods.
   Verify differentiability with a small autograd test.
5. Don't wire it to anything yet. Unit-test in isolation using
   hand-built traces (linear chain, branch without merge,
   concatenating merge, missing-parent error, key-collision error,
   OPL on a known paraxial geometry).

### Step 4 — Wire `OpticalTrace` into `Sequential`

Substeps:
1. Replace `SequentialData` with `OpticalTrace` as the token threaded
   through `Sequential.forward`. Keep `SequentialData` temporarily as a
   thin alias if helpful during transition; otherwise replace outright.
2. Add `OpticalTrace.empty(dim, dtype, device)` and a convention for
   the initial state (empty `nodes`, `bundles`, `tfs`). `Sequential`
   locally tracks the previous-step key as it iterates over its
   children:

   ```python
   prev_key: str | None = None
   for module_key, mod in self._modules.items():
       key = mod.trace_key or module_key
       parents = {prev_key} if prev_key is not None else set()
       trace = mod.extend_optical_trace(trace, key, parents)
       prev_key = key
   ```

   The first child (a light source) receives `parents=set()`.
3. Implement each element's `extend_optical_trace()` per its kind, following the
   per-element shapes in the Element protocol section. Each element
   passes `new_bundle` and `new_tf` to `trace.append(...)` as
   determined by its own semantics — surfaces pass both, apertures
   pass only a new bundle, spacers pass only a new tf, targets pass
   neither.
4. Element keying: use the `nn.Module._modules` key path
   (e.g. `"0.front"`) by default. Honor `trace_key` overrides exactly
   as today. `SubChain` namespaces children under its own key
   (`subchain.inner`).
5. Light sources: source nodes have `parents=()`. Their
   `OpticalTraceNode.tf` is the source placement frame. Their
   `bundle.P`/`V` are in the global frame.
6. Multi-source (concatenating merge): each source appends its own
   node with `parents=()`. The first downstream element after the
   second-or-later source sees a bundle that is the `cat` of all live
   sources, and its node has `parents=(source_A_key, source_B_key, ...)`.
   The cat'd bundle is fresh (introduced by that element), so the
   element's `new_bundle` is set normally — multi-parent semantics live
   entirely in the `parents` tuple, not in the bundle field.
7. Apertures: append a node with `ApertureRecord`, `new_tf=None`,
   `new_bundle=` the bundle with updated `valid`.
8. Focal point / image plane: append a node with the loss record;
   `new_bundle=None`, `new_tf=None`.

Tests:
- Existing forward-mode integration tests should pass (loss values
  unchanged within numerical tolerance).
- New tests: trace structure (node count, parent pointers, identity
  reuse for unchanged bundle/tf).

### Step 5 — Replace `ModelTrace`

Substeps:
1. Remove per-element `trace()` methods.
2. Remove `trace_model()` and the hook mechanism in
   `sequential/model_trace.py`.
3. Update the viewer / analysis modules to read from `OpticalTrace`
   instead of `ModelTrace`. The lookup helpers from step 2 grow
   parent-aware implementations on `OpticalTrace` so most call sites
   migrate without code change. Provide a `model_trace_compat()`
   shim only if a non-trivial external consumer needs it.
4. Delete `sequential/model_trace.py`.
5. Remove the `LightTargetOutput.surface_outputs` hack (the fake
   `SurfaceElementOutput` constructed in `FocalPoint.forward`
   previously needed for trace plumbing).

### Step 6 — OPL-based loss as a working example

Substeps:
1. Add a `wavefront_or_opl_loss` example (or extend an existing
   example) that uses `linear_path(trace, "source", "focal").opl()` as
   a loss term.
2. Verify gradients flow back to lens parameters.
3. Add a test that locks in the OPL value for a known geometry
   (paraxial lens with analytic OPL).

## Open sub-decisions to resolve during implementation

- **Aperture node and `new_bundle`**: an aperture produces a new bundle
  (because `valid` changed). It must therefore set `new_bundle` to the
  updated bundle, not `None`. Confirm this is the convention in step 3.
- **Multi-source merging semantics**: each source is its own node with
  `parents=set()`. The first consuming element receives
  `parents={source_a_key, source_b_key, ...}` listing every live source.
  The cat'd bundle is fresh — introduced by that element — so the
  consuming element's `new_bundle` is the cat result, and its
  `bundle_id` is its own key. Confirm during step 4 whether the cat
  happens inside that element's `extend_optical_trace()` (reading
  multiple parents from the trace) or via a helper called by
  `Sequential` before the consuming element fires.
- **`SampledVariable.cat` and non-shrinking bundles**: `cat` is still
  used on multi-source merges. Confirm the merge logic (validity,
  domain union) remains correct when bundles carry `valid` and `n`.
- **Naming**: `OpticalTrace` is locked. The container `Sequential`
  keeps its name. `linear_path` and `paths` names are tentative;
  finalize during step 3.

## Things explicitly out of scope

- Identifying merge nodes: multiple parents with the same source-aligned
  row layout combined such that row `k` is the same physical ray taking
  two paths (interferometric recombination).
- New composing containers (`Branch`, `Parallel`, `Merge`).
- Memory optimization for bundles (chunked evaluation, reduced-precision
  accumulators).
- Changes to surface kernels' numerical formulations beyond what is
  required to be NaN-safe.
- TIR `reflect` mode behavior changes.
