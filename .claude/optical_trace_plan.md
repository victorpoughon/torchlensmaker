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
  carries direct Python references to its input and output bundle and tf —
  no separate storage dicts, no string IDs. Python reference semantics
  ensure that nodes sharing an unchanged bundle or tf hold references to
  the same object without any tensor copying.
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
- Record types and module types are **decoupled**. `OpticalTraceNode` carries
  both the `record` (what data the element produced) and `module` (the
  instance that produced it). Consumers that care about the physical meaning
  dispatch on `type(node.module)`; consumers that care only about the data
  shape dispatch on `record.kind`.
- **Standardized `forward()` signatures**: every optical element's `forward`
  takes `(tf: Tf, rays: RayBundle)` (or just `tf` when rays are not
  consumed) and returns a single `ElementRecord` subclass — no tuples.
  The record carries all outputs of the call, including the output bundle
  where applicable.

## Data structures

### `ElementRecord` family

```python
class ElementRecord:
    kind: str  # discriminator; each subclass narrows to a Literal

@dataclass
class SourceRecord(ElementRecord):
    """Produced by LightSourceBase.forward(tf). Carries the emitted bundle."""
    output_rays: RayBundle
    kind: Literal["source"] = "source"

@dataclass
class SurfaceRecord(ElementRecord):
    """
    Produced by SurfaceElement.forward() — the pure geometry layer.
    Carries ray-surface collision data only; no output bundle.
    """
    t: BatchTensor           # P+tV are the collision points
    normals: BatchNDTensor   # unit normals at the collision points (global frame)
    valid: MaskTensor        # boolean mask for valid collisions
    points_local: BatchNDTensor  # collision points (surface frame)
    points_global: BatchNDTensor  # collision points (global frame)
    rsm: BatchTensor         # ray-surface minimum
    tf_surface: Tf           # surface transform
    tf_out: Tf               # output transform of the kinematic chain
    kind: Literal["surface"] = "surface"

@dataclass
class OpticalSurfaceRecord(ElementRecord):
    """
    Produced by OpticalSurfaceElement.forward(tf, rays) — the optical element
    layer. Wraps a SurfaceRecord from the child surface geometry and carries
    the output ray bundle produced by refraction, reflection, or aperture
    filtering.
    """
    output_rays: RayBundle
    surface: SurfaceRecord
    kind: Literal["optical_surface"] = "optical_surface"

@dataclass
class LightTargetRecord(ElementRecord):
    """Shared by focal points, image planes, and other light targets."""
    loss: ScalarTensor
    target_tf: Tf
    surface_outputs: SurfaceRecord  # temporary; removed in step 5
    kind: Literal["light_target"] = "light_target"

@dataclass
class KinematicRecord(ElementRecord):
    """Shared by all KinematicElement subclasses (e.g. Gap)."""
    tf_out: Tf
    kind: Literal["kinematic"] = "kinematic"
```

Records are the complete output of their corresponding `forward()` call.
Consumers dispatch on `record.kind` (or `isinstance`) for polymorphic
access. The `kind` field is placed last so positional fields keep their
default-free property.

The two-level surface design keeps concerns separated:
- `SurfaceRecord` is the geometry contract: "given rays and a transform,
  here is what hit what and where." It knows nothing about bundles.
- `OpticalSurfaceRecord` is the optical contract: "given an input bundle and
  a transform, here is the output bundle and the underlying collision data."

### `OpticalTraceNode`

```python
@dataclass
class OpticalTraceNode:
    key: str              # this node's own key (== dict key in trace.nodes)
    record: ElementRecord
    module: BaseModule    # the module instance that produced this node
    parents: set[str]     # set() for sources, {prev} for sequential
    bundle_in: RayBundle  # root_bundle for source nodes
    bundle_out: RayBundle # same object as bundle_in if no new bundle
    tf_in: Tf
    tf_out: Tf            # same object as tf_in if no new tf
```

- A node represents an *event* (an element fired). All four references are
  fully resolved at `append` time — no dict lookups or parent traversal.
- `module` is the live module instance. `type(node.module)` identifies which
  class fired; `node.module` gives direct access to parameters and state.
- `bundle_out is bundle_in` iff this node did not produce a new bundle.
  `bundle_out is not bundle_in` iff it did. Same pattern for tf. "Did
  element K change the bundle?" is one identity check.
- Source nodes receive `bundle_in = trace.root_bundle` (an empty
  `RayBundle`). All nodes receive either `trace.root_tf` or a preceding
  element's `tf_out` as their `tf_in`. No `None` values anywhere.
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
    root_bundle: RayBundle   # empty bundle; bundle_in for source nodes
    root_tf: Tf              # identity tf; tf_in for head-of-chain nodes

    # Resolution — direct attribute access, no dict lookup
    def bundle_in_at(self, key: str) -> RayBundle:
        return self.nodes[key].bundle_in
    def bundle_out_at(self, key: str) -> RayBundle:
        return self.nodes[key].bundle_out
    def tf_in_at(self, key: str) -> Tf:
        return self.nodes[key].tf_in
    def tf_out_at(self, key: str) -> Tf:
        return self.nodes[key].tf_out

    # Build
    def append(
        self,
        key: str,
        record: ElementRecord,
        module: BaseModule,
        parents: set[str],
        bundle_in: RayBundle,
        tf_in: Tf,
        new_bundle: RayBundle | None = None,   # None = share bundle_in as bundle_out
        new_tf: Tf | None = None,              # None = share tf_in as tf_out
    ) -> None: ...
        # bundle_out = new_bundle if new_bundle is not None else bundle_in
        # tf_out     = new_tf     if new_tf     is not None else tf_in
        # When parents has more than one element, new_bundle MUST NOT be
        # None — multi-parent bundle inheritance is ambiguous. Same for tf.

    # Queries
    @classmethod
    def empty(cls, dim: int, dtype: torch.dtype, device: torch.device) -> Self: ...
    def keys(self) -> list[str]: ...
    def is_linear(self) -> bool: ...
```

Trace-level invariants (cheap to assert in `__post_init__` or in tests):

- Every `node.bundle_in`, `node.bundle_out`, `node.tf_in`, `node.tf_out`
  is non-`None`. Source nodes use `trace.root_bundle` and `trace.root_tf`.
- `node.bundle_out is not node.bundle_in` iff this node introduced a new
  bundle (likewise for tf). No dangling references.
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

- **Kernel `forward`** (local, narrow, testable): uniform signature
  `forward(tf: Tf, rays: RayBundle)` (or `forward(tf: Tf)` when rays are
  not consumed), returns a **single `ElementRecord` subclass** carrying all
  outputs. No tuples.
- **`extend_optical_trace(trace, key, parents) -> trace`** (chain API):
  thin per-element method that reads its input from
  `trace.bundle_out_at(parent_key)` / `trace.tf_out_at(parent_key)`,
  calls its kernel, and appends a node. The caller (`Sequential`) is
  responsible for supplying `key` and the correct `parents` set.

Per-element shapes:

```python
# Optical surface (refractive or reflective) — new bundle and new tf
class OpticalSurfaceElement(BaseModule):
    def forward(self, tf: Tf, rays: RayBundle) -> OpticalSurfaceRecord:
        ...

    def extend_optical_trace(
        self, trace: OpticalTrace, key: str, parents: set[str]
    ) -> OpticalTrace:
        (parent_key,) = parents
        rays_in = trace.bundle_out_at(parent_key)
        tf_in = trace.tf_out_at(parent_key)
        record = self.forward(tf_in, rays_in)
        trace.append(
            key=key, record=record, module=self, parents=parents,
            bundle_in=rays_in, tf_in=tf_in,
            new_bundle=record.output_rays, new_tf=record.surface.tf_out,
        )
        return trace


# Aperture — new bundle (updated valid mask) but no new tf.
# Uses OpticalSurfaceRecord: apertures intersect rays with a surface geometry.
class Aperture(BaseModule):
    def forward(self, tf: Tf, rays: RayBundle) -> OpticalSurfaceRecord:
        ...

    def extend_optical_trace(
        self, trace: OpticalTrace, key: str, parents: set[str]
    ) -> OpticalTrace:
        (parent_key,) = parents
        rays_in = trace.bundle_out_at(parent_key)
        tf_in = trace.tf_out_at(parent_key)
        record = self.forward(tf_in, rays_in)
        trace.append(
            key=key, record=record, module=self, parents=parents,
            bundle_in=rays_in, tf_in=tf_in,
            new_bundle=record.output_rays, new_tf=None,
        )
        return trace


# Kinematic spacer — new tf, no new bundle.
class Gap(KinematicElement):
    def forward(self, tf: Tf) -> KinematicRecord:
        ...

    def extend_optical_trace(
        self, trace: OpticalTrace, key: str, parents: set[str]
    ) -> OpticalTrace:
        (parent_key,) = parents
        tf_in = trace.tf_out_at(parent_key)
        record = self.forward(tf_in)
        trace.append(
            key=key, record=record, module=self, parents=parents,
            bundle_in=trace.bundle_out_at(parent_key), tf_in=tf_in,
            new_bundle=None, new_tf=record.tf_out,
        )
        return trace


# Light target — no new bundle, no new tf; only a loss record.
class FocalPoint(BaseModule):
    def forward(self, tf: Tf, rays: RayBundle) -> LightTargetRecord:
        ...

    def extend_optical_trace(
        self, trace: OpticalTrace, key: str, parents: set[str]
    ) -> OpticalTrace:
        (parent_key,) = parents
        rays_in = trace.bundle_out_at(parent_key)
        tf_in = trace.tf_out_at(parent_key)
        record = self.forward(tf_in, rays_in)
        trace.append(
            key=key, record=record, module=self, parents=parents,
            bundle_in=rays_in, tf_in=tf_in,
        )
        return trace


# Light source — fresh bundle from tf input; no parent bundle.
class LightSourceBase(BaseModule):
    def forward(self, tf: Tf) -> SourceRecord:
        ...

    def extend_optical_trace(
        self, trace: OpticalTrace, key: str, parents: set[str]
    ) -> OpticalTrace:
        # Source nodes have parents=set(). tf_in is trace.root_tf or the
        # preceding kinematic element's tf_out. bundle_in is always
        # trace.root_bundle. When more than one source exists, downstream
        # elements consume a concatenating merge (see multi-source notes).
        tf_in = ...  # root_tf or preceding kinematic tf_out
        record = self.forward(tf_in)
        trace.append(
            key=key, record=record, module=self, parents=set(),
            bundle_in=trace.root_bundle, tf_in=tf_in,
            new_bundle=record.output_rays,
        )
        return trace
```

The kernel signatures are uniform: every `forward` takes `(tf, rays)` or
`(tf)` and returns a single record. `OpticalSurfaceRecord` wraps a
`SurfaceRecord` so collision data remains accessible. Kinematic elements
embed their output tf in `KinematicRecord.tf_out`. Sources embed their
output rays in `SourceRecord.output_rays`.

## Migration plan

Each step is independently shippable. After each step the codebase
should pass tests.

### Step 1 — Add `valid` and `n` to `RayBundle`

DONE.

### Step 2 — Introduce records and standardize `forward()` signatures

#### Step 2a — Define record types and standardize `forward()` signatures

This substep introduces the complete record vocabulary and standardizes
every optical element's `forward()` to the new convention. No changes to
`ModelTrace`, `SequentialData`, or the hook mechanism.

Files:
- `core/element_records.py` (new)
- `surfaces/surface_element.py` (return type updated)
- All concrete `SurfaceElement` subclasses (return type updated)
- `optical_surfaces/optical_surface.py` (new signature + `OpticalSurfaceRecord`)
- `optical_surfaces/refractive_surface.py`, `reflective_surface.py`,
  `aperture.py` (new signature)
- `kinematics/kinematics_elements.py` (new signature)
- `light_sources/light_sources_elements.py` (new signature)
- `light_targets/light_target.py`, `focal_point.py`, `image_plane.py`
  (new signature)

Record types to define:

- `SourceRecord(output_rays)` — source emits a fresh bundle.
- `SurfaceRecord(t, normals, valid, points_local, points_global, rsm,`
  `tf_surface, tf_out)` — pure geometry output of `SurfaceElement.forward`.
  Rename of the existing `tf_next` field to `tf_out`.
- `OpticalSurfaceRecord(output_rays, surface: SurfaceRecord)` — optical
  element output. Wraps the geometry record; carries the output bundle.
  Shared by refractive surfaces, reflective surfaces, and apertures.
- `LightTargetRecord(loss, target_tf, surface_outputs: SurfaceRecord)` —
  light target output. `surface_outputs` is temporary (removed in step 5).
- `KinematicRecord(tf_out)` — kinematic element output.

Standardized `forward()` conventions:

- `SurfaceElement.forward(tf, P, V) -> SurfaceRecord` — geometry layer,
  arg order updated.
- `OpticalSurfaceElement.forward(tf, rays) -> OpticalSurfaceRecord` — tf
  first, rays second, single record returned.
- `KinematicElement.forward(tf) -> KinematicRecord` — single record.
- `LightSourceBase.forward(tf) -> SourceRecord` — tf replaces the old
  `HomMatrix` arg; single record.
- `LightTarget.forward(tf, rays) -> LightTargetRecord` — tf first.

Tests:
- All existing forward / integration tests pass unchanged.
- `sequential()` methods updated internally to unpack fields from records.

#### Step 2b — Refactor `ModelTrace` to use `ElementRecord`

Pure refactor of the existing trace mode: parallel dicts collapse into a
`ModelTraceNode`-keyed `OrderedDict`. The hook mechanism and per-element
`trace()` methods stay; they call `add_node` instead of the old `add_*`
methods.

Files:
- `sequential/model_trace.py` (modified)

`ModelTrace` structure after this substep:

```python
@dataclass
class ModelTraceNode:
    record: ElementRecord
    module: BaseModule
    bundle_in: RayBundle
    bundle_out: RayBundle
    tf_in: Tf
    tf_out: Tf

@dataclass
class ModelTrace:
    dim: int
    nodes: OrderedDict[str, ModelTraceNode]
    root_bundle: RayBundle
    root_tf: Tf
```

`add_node(key, record, module, new_bundle, new_tf)` updates the linear
chain state (current bundle and tf) and appends a `ModelTraceNode`. All
four lookups are direct attribute accesses — no walking, no loops.

Guiding rule for record contents:

> A record carries only the data that is *intrinsic to the event the
> element produced*. It does not carry upstream context.

The produced bundle and `tf_out` live **on the record** in step 2 (`SourceRecord.output_rays`,
`OpticalSurfaceRecord.output_rays`, `KinematicRecord.tf_out`) and are
additionally stored on the node (`node.bundle_out`, `node.tf_out`) so that
`ModelTrace` helpers work without traversal.

Substeps:
1. Add `ModelTraceNode` dataclass. Update `ModelTrace`: replace
   `output_rays` with `bundles`, add `tfs`, `root_bundle`, `root_tf`.
2. Replace `ModelTrace`'s parallel dicts with
   `nodes: OrderedDict[str, ModelTraceNode]`.
3. `add_node(key, record, module, new_bundle, new_tf)` resolves all four
   references at call time using linear chain tracking.
4. Update each element's `trace()` method to call `add_node`, passing
   `new_bundle` and `new_tf` from the record.
5. Migrate the viewer and analysis modules onto `node.record` and
   `node.module`. All four lookup helpers become direct attribute reads.

Tests:
- Existing forward / trace integration tests pass unchanged.
- New tests: `kind`-discriminator invariants, lookup helpers correct
  across existing fixtures.

### Step 3 — Introduce `OpticalTrace` / `OpticalTraceNode` and derived views

Additive step: define the graph structure and OPL machinery, but do not
wire into `Sequential`.

Files added:
- `core/optical_trace.py`
- `core/optical_path.py`

Substeps:
1. Define `OpticalTraceNode` (`key`, `record`, `module`, `parents`,
   `bundle_in`, `bundle_out`, `tf_in`, `tf_out`) and `OpticalTrace`
   (`dim`, `dtype`, `device`, `nodes`, `root_bundle`, `root_tf`) with
   `append`, `bundle_in_at`, `bundle_out_at`, `tf_in_at`, `tf_out_at`,
   `is_linear`, `empty`. Enforce the trace-level invariants.
2. Define `OpticalPath` and `KinematicChain` derived views, plus
   `linear_path()`, `linear_kinematic_chain()`, and `paths()`.
3. Implement `OpticalPath.opl()` and `OpticalPath.segment_*` methods.
   Verify differentiability with a small autograd test.
4. Don't wire it to anything yet. Unit-test in isolation using
   hand-built traces (linear chain, branch without merge,
   concatenating merge, missing-parent error, key-collision error,
   OPL on a known paraxial geometry).

### Step 4 — Wire `OpticalTrace` into `Sequential`

Substeps:
1. Replace `SequentialData` with `OpticalTrace` as the token threaded
   through `Sequential.forward`. Keep `SequentialData` temporarily as a
   thin alias if helpful during transition; otherwise replace outright.
2. `Sequential` locally tracks the previous-step key as it iterates:

   ```python
   prev_key: str | None = None
   for module_key, mod in self._modules.items():
       key = mod.trace_key or module_key
       parents = {prev_key} if prev_key is not None else set()
       trace = mod.extend_optical_trace(trace, key, parents)
       prev_key = key
   ```

   The first child (a light source) receives `parents=set()`.
3. Implement each element's `extend_optical_trace()` per the element
   protocol. Each element reads `bundle_in` and `tf_in` from the parent
   node and passes `new_bundle` / `new_tf` from its record.
4. Element keying: use the `nn.Module._modules` key path
   (e.g. `"0.front"`) by default. Honor `trace_key` overrides exactly
   as today. `SubChain` namespaces children under its own key.
5. Multi-source (concatenating merge): each source appends its own node
   with `parents=set()`. The first downstream element after the
   second-or-later source sees `bundle_in = RayBundle.cat(...)` of all
   live source outputs, and has `parents={source_A_key, source_B_key, ...}`.

Tests:
- Existing forward-mode integration tests should pass (loss values
  unchanged within numerical tolerance).
- New tests: trace structure (node count, parent pointers, bundle/tf
  identity sharing for unchanged elements).

### Step 5 — Replace `ModelTrace`

Substeps:
1. Remove per-element `trace()` methods.
2. Remove `trace_model()` and the hook mechanism in
   `sequential/model_trace.py`.
3. Update the viewer / analysis modules to read from `OpticalTrace`
   instead of `ModelTrace`. The lookup helpers have the same signature
   in both so most call sites migrate without code change.
4. Delete `sequential/model_trace.py`.
5. Remove the `LightTargetRecord.surface_outputs` hack (the fake
   `SurfaceRecord` constructed in `FocalPoint.forward` previously needed
   for trace plumbing).

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
  updated bundle. Confirm this is the convention in step 3.
- **Multi-source merging semantics**: each source is its own node with
  `parents=set()`. The first consuming element receives
  `parents={source_a_key, source_b_key, ...}` listing every live source.
  The cat'd bundle is fresh, so the consuming element's
  `bundle_out is not bundle_in`. Confirm during step 4 whether the cat
  happens inside that element's `extend_optical_trace()` or via a helper
  called by `Sequential` before the consuming element fires.
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
