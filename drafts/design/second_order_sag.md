# Second-Order Derivatives for Sag and Implicit Functions

## Goal

Support analytical Hessians through the sag → lift → implicit function chain,
while avoiding computing them when not needed.

## Approach

Extend the existing `SagFunction` and `ImplicitFunction` types to carry an
optional Hessian as a third return element. No new TypeAliases are introduced.

## Modified Types

```python
# In sag_functions.py
# Before: (g, g_grad)
# After:  (g, g_grad, g_hess | None)
SagFunction: TypeAlias = Callable[
    [BatchNDTensor],
    tuple[BatchTensor, BatchNDTensor, BatchNDTensor | None]
]

# In implicit_solver.py
# Before: (F, F_grad)
# After:  (F, F_grad, F_hess | None)
ImplicitFunction: TypeAlias = Callable[
    [BatchNDTensor],
    tuple[BatchTensor, BatchNDTensor, BatchNDTensor | None]
]
```

`LiftFunction`, `ImplicitSolver`, and `DomainFunction` are **unchanged** — they
already refer to `SagFunction` and `ImplicitFunction`, so they inherit the new
semantics automatically.

## g_hess Shape

The Hessian of the sag function `g` with respect to its input coordinates:

- **2D**: `g_hess` is `d²g/dr²`, shape `(...)`
- **3D**: `g_hess` is the 2×2 Hessian of `G` with respect to `(y, z)`, shape `(..., 2, 2)`

For radially-symmetric 3D sag functions, where `G(y, z) = g(r)` with
`r = sqrt(y² + z²)`, the Hessian follows from the chain rule:

```
∂²G/∂y²    = g''(r) * y²/r²  +  g'(r) * z²/r³
∂²G/∂z²    = g''(r) * z²/r²  +  g'(r) * y²/r³
∂²G/∂y∂z   = g''(r) * yz/r²  -  g'(r) * yz/r³
```

## F_hess Shape

The Hessian of the implicit function `F` with respect to point coordinates:

- **2D**: 2×2 Hessian of `F` with respect to `(x, r)`, shape `(..., 2, 2)`
- **3D**: 3×3 Hessian of `F` with respect to `(x, y, z)`, shape `(..., 3, 3)`

For the raw 2D lift where `F(x, r) = nf * g(r/nf) - x`:

```
F_hess = [[0,  0             ],
          [0,  g''(r/nf)/nf  ]]
```

For the raw 3D lift where `F(x, y, z) = nf * G([y,z]/nf) - x`, the top-left
entry is zero and the lower-right 2×2 block is the Hessian of `G` (scaled by
`1/nf`).

## Changes Required

### sag_functions.py

- **Existing sag functions**: add `None` as third return value
- **New second-order sag functions**: compute and return `g_hess` analytically,
  following the same pattern as the existing gradient computations

### sag_functions.py — lift functions

- **Existing lift functions**: unpack `g, g_grad, g_hess = sag(...)`, return
  `(f, f_grad, None)`
- **New second-order lift functions**: compute `F_hess` from `g_hess`
  analytically and return `(f, f_grad, f_hess)`

### implicit_solver.py

- **`implicit_solver_newton`**: unpack with `_` to ignore hessian:
  `F, F_grad, _ = implicit_function(points)`

## Summary

| Item | Action |
|---|---|
| `SagFunction` | Modified — optional 3rd return element `g_hess` |
| `ImplicitFunction` | Modified — optional 3rd return element `F_hess` |
| `LiftFunction` | Unchanged |
| `ImplicitSolver` | Unchanged |
| `DomainFunction` | Unchanged |
| Existing sag functions | Add `, None` to return |
| New second-order sag functions | Return analytical `g_hess` |
| Existing lift functions | Unpack 3-tuple, pass through `None` |
| New second-order lift functions | Compute `F_hess` from `g_hess` |
| `implicit_solver_newton` | Ignore hessian with `_` |
