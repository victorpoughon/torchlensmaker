# Model Export Format & Library Design

## Overview

This document describes the design of a PyTorch-based modeling library that lets users compose models from a set of provided `nn.Module` building blocks, and serialize composed models to a human-readable, schema-validated YAML format inspired by Static Single Assignment (SSA) form.

The export format is structural: it captures the topology of the model (which ops, in what order, with what wiring) and the parameter values of each op (including learned weight tensors). The format is designed to be the on-disk source of truth — language-agnostic, diffable, and amenable to tooling — while the PyTorch library provides the authoring frontend.

This document covers the export side. Importing serialized models back into executable PyTorch is out of scope for v1, but the design is intended not to preclude it.

## Goals

- Users author models in idiomatic PyTorch by composing `nn.Module` subclasses provided by the library.
- Composed models export to a single self-contained YAML file capturing structure and parameters.
- The export format is schema-defined (Pydantic), validated at every boundary, and human-readable.
- Composition is hierarchical — composites contain children — but the on-disk form is a *flat* list of nodes with explicit parent references, avoiding deep YAML nesting.
- Each module's `export()` is independently callable and returns a fragment that parents can wire into a larger graph.

## Non-goals (for v1)

- Importing YAML back into executable PyTorch (future work).
- Strict shape/dtype validation in the export format itself (deferred to runtime).
- Multiple entry points per module — each module has a single `forward()` interface.
- General-purpose tracing of arbitrary PyTorch code. The library's modules are explicitly `export()`-aware; we do not aim to export arbitrary user code.

## Two orthogonal relationships

The format expresses two distinct relationships between nodes, and it is important to keep them separate:

- **Dataflow** — which values feed which inputs. A node has zero or more dataflow inputs, each referencing the named output of some earlier node. This is the SSA dimension.
- **Containment** — which composite a node belongs to. A node has zero or one structural parent. Roots have none; everything else is contained in exactly one composite. This is the hierarchy dimension.

These dimensions are independent. A node's dataflow inputs are unrelated to its structural parent, and the rules governing each are separate.

## Background: SSA-inspired representation

Static Single Assignment is a compiler IR convention where every named value is defined exactly once. Operations consume previously-defined values (their inputs) and produce new named values (their outputs). The textual order of operations gives a topological sort of the dataflow graph.

This maps cleanly onto our composition model:

- A **module instance** is one node in the graph.
- Its **outputs** are SSA values it defines, addressed by name (a node may have multiple).
- Its **inputs** reference SSA values defined by earlier nodes, by `(node_id, output_name)`.
- Its **parameters** carry both static configuration values and learned tensor weights, declared by the op's registry entry.

A composite module (e.g. `Sequential`) is itself a node, with an interface (declared inputs/outputs). Its children are separate nodes that name it as their structural parent. The composite's outputs are *bound* to internal values via explicit output bindings on the composite node.

## The op registry

The library maintains a registry of known operations. Each entry declares:

- The op's **name** as a string (e.g. `"Linear"`, `"ReLU"`, `"Sequential"`). String-only, no namespacing or versioning in v1 — we accept this as a forward-compatibility risk to keep the format simple. If versioning becomes necessary, an optional version field can be added without breaking existing files.
- The op's **input schema**: ordered list of named inputs.
- The op's **output schema**: ordered list of named outputs.
- The op's **parameter schema**: named parameters with their types. A parameter may be a scalar (e.g. `int`, `bool`), a tensor (with declared dtype/shape conventions), or another simple value the op needs.
- A reference to the corresponding `nn.Module` class.

Registration is done via a decorator on the module class:

```python
@register_op("Linear")
class Linear(ExportableModule):
    inputs = ["x"]
    outputs = ["y"]
    parameters = {
        "weight": Tensor,
        "bias": Tensor,  # optional, depending on configuration
        "in_features": int,
        "out_features": int,
    }
    ...
```

The registry is the contract between the export format and the runtime: any op name appearing in a YAML file must resolve to a registered op, and the inputs/outputs/parameters declared on that op define what's valid in the YAML.

## Parameters

Parameters belong to a node. There is a single notion of parameter — covering both static configuration (`in_features=64`) and learned tensor weights (`weight=<tensor>`) — distinguished only by their type and by a `learnable` flag (mapping to PyTorch's `requires_grad`). The op's parameter *schema* (which parameters exist and what types they have) lives in the registry; the export carries the *values*.

Tensor parameters are serialized inline using base64. This makes the file self-contained at the cost of file size for large models.

```python
class TensorValue(BaseModel):
    dtype: str            # e.g. "float32"
    shape: list[int]
    data: str             # base64-encoded raw bytes

class ParameterValue(BaseModel):
    value: TensorValue | int | float | bool | str | None
    learnable: bool = False   # only meaningful for tensor values
```

Parameters do not bubble up to a top-level dictionary; they are local to the node that uses them. Hierarchical addressing (`encoder.block_0.linear_0.parameters.weight`) is a property of the dotted id system described below, not a property of the parameter representation.

## Node ids and the flat list

The exported model is a flat list of nodes. Each node has:

- A globally unique **dotted id** that encodes its position in the structural hierarchy (`encoder.block_0.linear_0`).
- An explicit **parent field** giving the dotted id of its containing composite (or `null` for root-level nodes).

Both representations are kept. The dotted id is the canonical identifier and is unique across the whole model; the parent field makes hierarchy reconstruction trivial without parsing ids. They are required to be consistent: a node's parent must be the dotted prefix of its id with the last segment removed (`encoder.block_0` for `encoder.block_0.linear_0`). Validation enforces this.

The flat list is in **topological order** with respect to *both* relationships:

- A node's dataflow inputs reference only nodes earlier in the list.
- A node's structural parent (if any) appears earlier in the list.

The combined ordering is straightforward to produce: emit the root-level nodes in topological order, and within each composite, emit its children in topological order immediately after the composite's own entry. A composite's outputs become available after its last child has been emitted; sibling nodes that consume them appear later in the flat list than the composite's final child.

## Dataflow scope rules

A node's dataflow inputs may reference:

- Outputs of earlier nodes within the same containing composite (siblings).
- The declared inputs of the containing composite (addressed via a sentinel — see below).
- The outputs of the containing composite itself, after the composite has been "closed" — i.e., other nodes at the *parent* scope can reference the composite as a whole.

A node may **not** freely reach into another scope. In particular:

- Inner nodes cannot reference values from enclosing scopes except through the composite's declared inputs. This keeps composites self-contained units.
- Nodes outside a composite cannot reference values internal to it; only the composite's declared outputs are visible from outside.

This mirrors region semantics in MLIR and LLVM and is what makes composites genuinely modular.

## Value references and the input sentinel

A `ValueRef` always names both the producing node and the specific output:

```python
class ValueRef(BaseModel):
    node: str       # dotted node id, OR the sentinel "%input"
    output: str     # name of the output (or input, when node == "%input")
```

To reference one of the enclosing composite's declared inputs, a node uses the sentinel `node = "%input"` together with the input name. The `%input` sentinel is *scoped*: it always refers to the inputs of the immediately enclosing composite (or to the top-level model's inputs if the node is at root). There is no syntax for reaching past one level of scope — values must be threaded through composite interfaces explicitly.

## Pydantic schema

```python
class ExportedModel(BaseModel):
    format_version: str            # e.g. "0.1"
    inputs: list[str]              # top-level model input names
    outputs: list[ValueRef]        # top-level outputs, bound to internal values
    nodes: list[Node]              # flat list, topologically ordered

class Node(BaseModel):
    id: str                        # globally unique dotted id
    parent: str | None             # dotted id of containing composite, or null
    op: str                        # registry name
    inputs: dict[str, ValueRef]    # input_name -> ValueRef
    outputs: list[str]             # names of outputs this node produces
    parameters: dict[str, ParameterValue]
    output_bindings: dict[str, ValueRef] | None
        # present only on composite nodes:
        # maps each declared output name to an internal ValueRef

class ValueRef(BaseModel):
    node: str                      # dotted node id, or "%input"
    output: str                    # output (or input) name

class TensorValue(BaseModel):
    dtype: str
    shape: list[int]
    data: str                      # base64-encoded bytes

class ParameterValue(BaseModel):
    value: TensorValue | int | float | bool | str | None
    learnable: bool = False
```

### Composite output bindings

Composites declare their outputs in `outputs`, and *bind* each declared output to an internal `ValueRef` via `output_bindings`. This keeps the composite's interface and its internal wiring colocated on the same node.

For leaf nodes, `output_bindings` is `null`. For composites, it must contain exactly one entry per declared output, and each binding must reference a `ValueRef` resolvable from inside the composite (i.e., to one of the composite's children, or to the composite's own `%input`).

### Example

A `Sequential(Linear(8, 16), ReLU(), Linear(16, 4))` exports as:

```yaml
format_version: "0.1"
inputs: [x]
outputs:
  - {node: root, output: y}
nodes:
  - id: root
    parent: null
    op: Sequential
    inputs:
      x: {node: "%input", output: x}
    outputs: [y]
    parameters: {}
    output_bindings:
      y: {node: root.linear_2, output: y}

  - id: root.linear_0
    parent: root
    op: Linear
    inputs:
      x: {node: "%input", output: x}
    outputs: [y]
    parameters:
      in_features: {value: 8, learnable: false}
      out_features: {value: 16, learnable: false}
      weight:
        value: {dtype: float32, shape: [16, 8], data: "<base64>"}
        learnable: true
      bias:
        value: {dtype: float32, shape: [16], data: "<base64>"}
        learnable: true
    output_bindings: null

  - id: root.relu_1
    parent: root
    op: ReLU
    inputs:
      x: {node: root.linear_0, output: y}
    outputs: [y]
    parameters: {}
    output_bindings: null

  - id: root.linear_2
    parent: root
    op: Linear
    inputs:
      x: {node: root.relu_1, output: y}
    outputs: [y]
    parameters:
      in_features: {value: 16, learnable: false}
      out_features: {value: 4, learnable: false}
      weight:
        value: {dtype: float32, shape: [4, 16], data: "<base64>"}
        learnable: true
      bias:
        value: {dtype: float32, shape: [4], data: "<base64>"}
        learnable: true
    output_bindings: null
```

Note that `root.linear_0` and `root.relu_1` reference `%input` from *their* enclosing scope (which is `root`, the Sequential). The `%input` sentinel resolves relative to the node's containing composite, not globally.

## The library: `ExportableModule` and `export()`

The library provides an `ExportableModule` base class extending `nn.Module`. Every op registered with the library subclasses it and implements `export()`.

### The `export()` contract

`export()` is **standalone-callable on any module** and returns an export *fragment* — a list of nodes describing this module and (if it is a composite) all of its children.

```python
class ExportFragment(BaseModel):
    """The result of exporting a single module."""
    root_id: str                    # dotted id of the module's own node
    nodes: list[Node]               # this module's node + descendants, topo-ordered

class ExportableModule(nn.Module):
    def export(self, *, id: str, input_refs: dict[str, ValueRef]) -> ExportFragment:
        """
        Export this module.

        Args:
            id: the dotted id this module's node should have. The parent
                determines this; for top-level export it is "root".
            input_refs: a mapping from this module's declared input names
                        to ValueRefs resolvable in the parent scope.

        Returns:
            An ExportFragment whose first node has id `id`, with `parent`
            set appropriately, followed by any descendant nodes.
        """
        ...
```

The contract is symmetric: the parent decides the node's `id` (because uniqueness and dotted-prefix correctness are parent-scope concerns) and supplies the wiring. The child supplies everything internal: its `op` name, declared `outputs`, `parameters`, `output_bindings` (if composite), and any descendants.

### Leaf module example

```python
@register_op("Linear")
class Linear(ExportableModule):
    inputs = ["x"]
    outputs = ["y"]

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return self.linear(x)

    def export(self, *, id, input_refs):
        params = {
            "in_features": ParameterValue(value=self.in_features, learnable=False),
            "out_features": ParameterValue(value=self.out_features, learnable=False),
            "weight": ParameterValue(
                value=tensor_to_value(self.linear.weight),
                learnable=self.linear.weight.requires_grad,
            ),
        }
        if self.linear.bias is not None:
            params["bias"] = ParameterValue(
                value=tensor_to_value(self.linear.bias),
                learnable=self.linear.bias.requires_grad,
            )

        node = Node(
            id=id,
            parent=parent_of(id),    # dotted prefix or None
            op="Linear",
            inputs=input_refs,
            outputs=["y"],
            parameters=params,
            output_bindings=None,
        )
        return ExportFragment(root_id=id, nodes=[node])
```

### Composite module example

A composite walks its children, assigns dotted ids, threads `ValueRef`s through them, collects all child nodes into the flat list, and finally emits its own composite node with the appropriate `output_bindings`:

```python
@register_op("Sequential")
class Sequential(ExportableModule):
    inputs = ["x"]
    outputs = ["y"]

    def __init__(self, *children: ExportableModule):
        super().__init__()
        self.children_modules = nn.ModuleList(children)

    def forward(self, x):
        for child in self.children_modules:
            x = child(x)
        return x

    def export(self, *, id, input_refs):
        all_nodes: list[Node] = []
        # Inside this composite, "%input" refers to our own declared inputs.
        # The first child reads from %input.x; subsequent children read from
        # the previous child's output.
        current_ref = ValueRef(node="%input", output="x")
        last_child_id: str | None = None
        last_child_output: str | None = None

        for i, child in enumerate(self.children_modules):
            child_short = f"{type(child).__name__.lower()}_{i}"
            child_id = f"{id}.{child_short}"
            child_input_name = child.inputs[0]

            fragment = child.export(
                id=child_id,
                input_refs={child_input_name: current_ref},
            )
            all_nodes.extend(fragment.nodes)

            # Use the child's first declared output as the threaded value.
            primary_output = fragment.nodes[0].outputs[0]
            current_ref = ValueRef(node=child_id, output=primary_output)
            last_child_id = child_id
            last_child_output = primary_output

        # Emit the composite's own node first in topological order:
        # composite-entry, then its children, then later siblings.
        composite_node = Node(
            id=id,
            parent=parent_of(id),
            op="Sequential",
            inputs=input_refs,
            outputs=["y"],
            parameters={},
            output_bindings={
                "y": ValueRef(node=last_child_id, output=last_child_output),
            },
        )
        return ExportFragment(
            root_id=id,
            nodes=[composite_node, *all_nodes],
        )
```

The ordering convention here is: the composite's own node comes first in its fragment, followed by all descendants in topological order. When the parent splices fragments together, this naturally produces a globally topological flat list.

### Top-level entry point

```python
def export_model(module: ExportableModule) -> ExportedModel:
    fragment = module.export(
        id="root",
        input_refs={
            name: ValueRef(node="%input", output=name)
            for name in module.inputs
        },
    )
    # Bind the top-level outputs to the root composite's outputs.
    top_outputs = [
        ValueRef(node="root", output=out_name)
        for out_name in module.outputs
    ]
    return ExportedModel(
        format_version="0.1",
        inputs=list(module.inputs),
        outputs=top_outputs,
        nodes=fragment.nodes,
    )
```

The sentinel `"%input"` at the top level denotes the user-supplied inputs to the model itself.

## Validation

Validation occurs at three points, in this order:

1. **Schema validation (Pydantic).** When loading a YAML file, Pydantic enforces the structural schema: required fields, types, nested model shapes. This catches malformed files immediately.

2. **Graph validation (custom).** After Pydantic parses the file, a separate pass checks SSA discipline and structural consistency:
   - Every node's `id` is globally unique.
   - Every node's `parent` is either `null` or a dotted prefix of its `id` matching an earlier node in the list.
   - Every node's `op` resolves in the registry, and its declared `inputs`/`outputs`/`parameters` match the registry schema.
   - Every `ValueRef` in `inputs` resolves to a node earlier in the flat list, in the same scope, addressing a declared output of that node — *or* uses `%input` referencing a declared input of the immediately enclosing composite.
   - Every composite node has `output_bindings` covering its declared outputs, with each binding resolvable from inside the composite.
   - No `ValueRef` reaches across scope boundaries (no inner-to-outer references except via `%input`, no outer-to-inner references except to the composite's own outputs).
   - Top-level `outputs` resolve to root-level nodes' declared outputs.

3. **Runtime validation.** Shape/dtype mismatches and operational errors surface during execution, not at export or load time. This is a deliberate v1 simplification.

Graph validation produces actionable error messages with node ids and (where possible) source locations from the YAML file. We use `ruamel.yaml` for loading so that line numbers can be preserved and surfaced in errors.

## Editor tooling

Pydantic's `ExportedModel.model_json_schema()` produces a JSON Schema that we ship with the library. Users can configure the VS Code YAML extension (or any JSON-Schema-aware editor) to use it, getting autocomplete, inline validation, and hover documentation while inspecting or hand-editing exported files.

## Open questions / future work

- **Versioning.** The op-name-only registry is intentionally simple. If schema evolution becomes painful, we will add an optional `version` field on nodes and on the registry, and a small compatibility layer for migrations.
- **External weight storage.** Inline base64 is convenient but bloats files for large models. A future option could allow tensor parameters to be stored in a sidecar `.safetensors` file referenced by path.
- **Import / round-trip.** A loader that consumes the YAML and reconstructs an executable PyTorch model is the natural next step. The current design does not preclude it: every op is registered with its `nn.Module` class, the SSA structure is unambiguous, and the flat list with explicit parents makes hierarchy reconstruction mechanical.
- **Dynamic structure.** Modules whose graph depends on input data (e.g. variable-length recursion) are not handled. The export format assumes static composition.
- **Multiple entry points.** Currently each module has one `forward()`. If multi-method modules become necessary, the schema would extend `Node` with a per-method input/output schema.
- **Shared subgraphs.** If two parts of a model need to share the same op instance (e.g., tied embeddings), the current format requires them to be modeled as separate nodes that happen to carry identical parameter values. A future extension could add a notion of references / aliases for true sharing.
