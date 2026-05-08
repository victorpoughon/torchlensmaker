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
- `OpticalTrace` is a flat `OrderedDict[str, OpticalTraceNode]`. Each node
  carries an `ElementRecord`, parent pointers, and **optional** `bundle`
  and `tf` fields (`None` means inherited from parents).
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
class ElementRecord: ...

@dataclass
class SourceRecord(ElementRecord): ...

@dataclass
class CollisionRecord(ElementRecord):
    t: BatchTensor
    normals: BatchNDTensor
    valid: MaskTensor
    points_local: BatchNDTensor
    points_global: BatchNDTensor
    rsm: BatchTensor
    tf_surface: Tf

@dataclass
class ApertureRecord(ElementRecord):
    valid_update: MaskTensor

@dataclass
class FocalRecord(ElementRecord):
    loss: ScalarTensor
    target_tf: Tf

# Plus: ImagePlaneRecord, GapRecord (kinematic spacer), etc., as elements
# are migrated.
```

Records replace the parallel dicts on `ModelTrace`. They are the only
polymorphic per-element data; consumers dispatch on type.

### `OpticalTraceNode`

```python
@dataclass
class OpticalTraceNode:
    record: ElementRecord
    parents: tuple[str, ...]      # () for sources, (prev,) for sequential
    bundle: RayBundle | None      # None = inherited from parents
    tf: Tf | None                 # None = inherited from parents
```

- A node represents an *event* (an element fired). The bundle and tf are
  state the event may or may not update.
- `bundle is None` means: ask the parents. Resolution walks parent
  pointers until a non-`None` value is found. (For the locked
  no-true-merge scope, parents has at most 1 element, so resolution is
  unambiguous.)

### `OpticalTrace`

```python
@dataclass
class OpticalTrace:
    nodes: OrderedDict[str, OpticalTraceNode]
    frontier: str | None          # latest node added (for live build)

    # Resolution
    def bundle_at(self, key: str) -> RayBundle: ...
    def tf_at(self, key: str) -> Tf: ...

    # Build
    def append(
        self,
        key: str,
        record: ElementRecord,
        parents: tuple[str, ...],
        new_bundle: RayBundle | None,
        new_tf: Tf | None,
    ) -> None: ...

    # Queries
    def keys(self) -> list[str]: ...
    def is_linear(self) -> bool: ...
```

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
- **`extend_optical_trace(trace) -> trace`** (chain API): thin per-element method that
  reads its input from the trace, calls its kernel, and appends a node
  with `new_bundle` and `new_tf` set to exactly what the element type
  produces. No identity comparison, no implicit dedup — each element
  knows its own semantics.

Per-element shapes:

```python
# Refractive / reflective surface — produces both a new bundle and a new tf
class OpticalSurfaceElement(BaseModule):
    def forward(self, rays: RayBundle, tf: Tf) -> tuple[RayBundle, CollisionRecord]:
        ...

    def extend_optical_trace(self, trace: OpticalTrace) -> OpticalTrace:
        rays_in = trace.bundle_at(trace.frontier)
        tf_in = trace.tf_at(trace.frontier)
        new_rays, record = self.forward(rays_in, tf_in)
        trace.append(
            key=self.qualified_name,
            record=record,
            parents=(trace.frontier,),
            new_bundle=new_rays,
            new_tf=record.tf_next,
        )
        return trace


# Aperture — produces a new bundle (updated valid) but no new tf
class Aperture(BaseModule):
    def forward(self, rays: RayBundle, tf: Tf) -> tuple[RayBundle, ApertureRecord]:
        ...

    def extend_optical_trace(self, trace: OpticalTrace) -> OpticalTrace:
        rays_in = trace.bundle_at(trace.frontier)
        tf_in = trace.tf_at(trace.frontier)
        new_rays, record = self.forward(rays_in, tf_in)
        trace.append(
            key=self.qualified_name,
            record=record,
            parents=(trace.frontier,),
            new_bundle=new_rays,
            new_tf=None,
        )
        return trace


# Kinematic spacer — produces a new tf but no new bundle
class Gap(BaseModule):
    def forward(self, tf: Tf) -> tuple[Tf, GapRecord]:
        ...

    def extend_optical_trace(self, trace: OpticalTrace) -> OpticalTrace:
        tf_in = trace.tf_at(trace.frontier)
        new_tf, record = self.forward(tf_in)
        trace.append(
            key=self.qualified_name,
            record=record,
            parents=(trace.frontier,),
            new_bundle=None,
            new_tf=new_tf,
        )
        return trace


# Light target (e.g. focal point) — produces neither a new bundle nor a new tf,
# only a record carrying the loss
class FocalPoint(BaseModule):
    def forward(self, rays: RayBundle, tf: Tf) -> FocalRecord:
        ...

    def extend_optical_trace(self, trace: OpticalTrace) -> OpticalTrace:
        rays_in = trace.bundle_at(trace.frontier)
        tf_in = trace.tf_at(trace.frontier)
        record = self.forward(rays_in, tf_in)
        trace.append(
            key=self.qualified_name,
            record=record,
            parents=(trace.frontier,),
            new_bundle=None,
            new_tf=None,
        )
        return trace


# Light source — produces a fresh bundle, no parents
class LightSourceBase(BaseModule):
    def forward(self, tf: Tf) -> tuple[RayBundle, SourceRecord]:
        ...

    def extend_optical_trace(self, trace: OpticalTrace) -> OpticalTrace:
        # Source nodes have parents=(); their tf is the source placement.
        # When more than one source exists in a model, downstream elements
        # consume a concatenating merge of all live sources (see
        # multi-source notes below).
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

### Step 2 — Introduce `OpticalTrace` and `ElementRecord` family

Files added:
- `core/optical_trace.py` (or `sequential/optical_trace.py`)
- `core/element_records.py`
- `core/optical_path.py`

Substeps:
1. Define `ElementRecord` base class and concrete records:
   `SourceRecord`, `CollisionRecord`, `ApertureRecord`, `FocalRecord`,
   `ImagePlaneRecord`, `GapRecord`. Mirror current `ModelTrace`
   information so nothing is lost.
2. Define `OpticalTraceNode` and `OpticalTrace` with `append`,
   `bundle_at`, `tf_at`, `is_linear`, plus a `frontier` pointer.
3. Define `OpticalPath` and `KinematicChain` derived views, plus
   `linear_path()`, `linear_kinematic_chain()`, and `paths()`.
4. Implement `OpticalPath.opl()` and `OpticalPath.segment_*` methods.
   Verify differentiability with a small autograd test.
5. Don't wire it to anything yet. Unit-test in isolation using
   hand-built traces.

### Step 3 — Wire `OpticalTrace` into `Sequential`

Substeps:
1. Replace `SequentialData` with `OpticalTrace` as the token threaded
   through `Sequential.forward`. Keep `SequentialData` temporarily as a
   thin alias if helpful during transition; otherwise replace outright.
2. Add `OpticalTrace.empty(dim, dtype, device)` and a convention for
   the initial state (no nodes, `frontier=None`).
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

### Step 4 — Replace `ModelTrace`

Substeps:
1. Remove per-element `trace()` methods.
2. Remove `trace_model()` and the hook mechanism in
   `sequential/model_trace.py`.
3. Update the viewer / analysis modules to read from `OpticalTrace`
   instead of `ModelTrace`. Provide a `model_trace_compat()` shim only
   if a non-trivial external consumer needs it; otherwise migrate
   call sites directly.
4. Delete `sequential/model_trace.py`.
5. Remove the `LightTargetOutput.surface_outputs` hack (the fake
   `SurfaceElementOutput` constructed in `FocalPoint.forward`
   previously needed for trace plumbing).

### Step 5 — OPL-based loss as a working example

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
  updated bundle, not `None`. Confirm this is the convention in step 1.
- **Multi-source merging semantics**: each source is its own node with
  `parents=()`. `Sequential` cats live source bundles when feeding the
  next element. The first consuming element's node has `parents` listing
  every contributing source. Confirm during step 3 whether the cat
  happens inside that element's `extend_optical_trace()` (reading multiple frontiers
  from the trace) or via a synthetic merge node added by `Sequential`
  before the consuming element fires.
- **Empty `OpticalTrace` and the first input**: when `frontier is None`
  and the first element runs, where does its input come from? Light
  sources are special-cased to need no input bundle. Document in
  `Sequential.forward`.
- **`SampledVariable.cat` and non-shrinking bundles**: `cat` is still
  used on multi-source merges. Confirm the merge logic (validity,
  domain union) remains correct when bundles carry `valid` and `n`.
- **Naming**: `OpticalTrace` is locked. The container `Sequential`
  keeps its name. `linear_path` and `paths` names are tentative;
  finalize during step 2.

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
