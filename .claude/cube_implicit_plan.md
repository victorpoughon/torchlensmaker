# Plan: Add 3D Cube Implicit SDF to torchimplicit

## Context

Add a signed distance function (SDF) for a cube centered at the origin with a single side-length parameter. F > 0 outside, F = 0 on the surface, F < 0 inside.

## Math

For half-side `a = s/2`, define face distances `q = (|x|−a, |y|−a, |z|−a)`:

```
outer = ||max(q, 0)||          # Euclidean dist to nearest surface point (0 inside)
inner = clamp(max(qx,qy,qz), max=0)   # closest face dist, ≤ 0 (0 outside)
F = outer + inner              # signed distance
```

**Gradient**

*Outside* (`outer > 0`): let `gxc = sign(x)·max(qx,0)` etc. (unnormalized outer-grad components):
```
grad = [gxc, gyc, gzc] / outer
```

*Inside* (`outer = 0`): `F = max(qx,qy,qz)`, gradient is a unit face normal:
```
dominant face x: grad = [sign(x), 0, 0]
dominant face y: grad = [0, sign(y), 0]
dominant face z: grad = [0, 0, sign(z)]
```

**Hessian**

*Outside* (`outer > 0`): let `di = [qi > 0]` (1 if that face contributes to outer):
```
H[i,j] = (di · δij / outer) − (gic · gjc / outer³)
```
This reduces to: 0 in face regions (linear), circle-curvature in edge regions, sphere-curvature in corner regions.

*Inside* (`outer = 0`): H = 0 (piecewise linear function).

**Non-smooth regions** (test must avoid):
- Cube surface edges/corners (where any two `qi = 0`)
- Interior medial axis: where `|ax − ay|`, `|ax − az|`, or `|ay − az|` ≈ 0 (two face distances tie)

## Files to Create/Modify

### 1. Create `torchimplicit/src/torchimplicit/implicit_cube.py`

```python
import torch

from torchimplicit.domain import total_domain
from torchimplicit.math import safe_div, safe_sign
from torchimplicit.registry import example_scalar, register_implicit_function
from torchimplicit.types import ImplicitFunction, ImplicitResult


def implicit_cube_3d(points, params, *, order):
    """
    Signed distance function for a cube centered at the origin.

        q = (|x| − a, |y| − a, |z| − a)   where a = s / 2
        F = ||max(q, 0)|| + min(max(qx, qy, qz), 0)

    Params:
        s (scalar): side length

    Note: not differentiable at cube surface edges/corners and interior medial axis.
    """
    assert params.shape == (1,), f"cube_3d expects (1,), got {params.shape}"
    a = params[0] / 2

    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    ax, ay, az = x.abs(), y.abs(), z.abs()
    qx, qy, qz = ax - a, ay - a, az - a

    qxc = qx.clamp(min=0)
    qyc = qy.clamp(min=0)
    qzc = qz.clamp(min=0)

    outer = (qxc**2 + qyc**2 + qzc**2).sqrt()
    inner = torch.maximum(torch.maximum(qx, qy), qz).clamp(max=0)

    F = outer + inner

    # pre-compute unnormalized outer-grad components (needed for grad and hess)
    sx, sy, sz = safe_sign(x), safe_sign(y), safe_sign(z)
    gxc = sx * qxc
    gyc = sy * qyc
    gzc = sz * qzc

    grad = None
    if order >= 1:
        inv_outer = safe_div(torch.ones_like(outer), outer)

        x_dom = (qx >= qy) & (qx >= qz)
        y_dom = (qy > qx) & (qy >= qz)
        zeros = torch.zeros_like(x)
        outside = outer > 0

        grad_x = torch.where(outside, gxc * inv_outer,
                              torch.where(x_dom, sx, zeros))
        grad_y = torch.where(outside, gyc * inv_outer,
                              torch.where(y_dom, sy, zeros))
        grad_z = torch.where(outside, gzc * inv_outer,
                              torch.where(x_dom | y_dom, zeros, sz))

        grad = torch.stack([grad_x, grad_y, grad_z], dim=-1)

    hess = None
    if order >= 2:
        zeros = torch.zeros_like(x)
        outside = outer > 0
        inv_outer = safe_div(torch.ones_like(outer), outer)
        inv_outer3 = safe_div(torch.ones_like(outer), outer**3)

        dx = qx.gt(0).to(x.dtype)
        dy = qy.gt(0).to(x.dtype)
        dz = qz.gt(0).to(x.dtype)

        H_xx = torch.where(outside, dx * inv_outer - gxc**2 * inv_outer3, zeros)
        H_yy = torch.where(outside, dy * inv_outer - gyc**2 * inv_outer3, zeros)
        H_zz = torch.where(outside, dz * inv_outer - gzc**2 * inv_outer3, zeros)
        H_xy = torch.where(outside, -gxc * gyc * inv_outer3, zeros)
        H_xz = torch.where(outside, -gxc * gzc * inv_outer3, zeros)
        H_yz = torch.where(outside, -gyc * gzc * inv_outer3, zeros)

        hess = torch.stack([
            torch.stack([H_xx, H_xy, H_xz], dim=-1),
            torch.stack([H_xy, H_yy, H_yz], dim=-1),
            torch.stack([H_xz, H_yz, H_zz], dim=-1),
        ], dim=-2)

    return ImplicitResult(F, grad, hess)


cube_3d = ImplicitFunction(
    name="cube_3d",
    dim=3,
    func=implicit_cube_3d,
    n_params=1,
    param_names=("s",),
    example_params=example_scalar(5.0),
    domain=total_domain,
)
register_implicit_function(cube_3d)
```

### 2. Update `torchimplicit/src/torchimplicit/__init__.py`

```python
from .implicit_cube import cube_3d
# add "cube_3d" to __all__
```

### 3. Update `torchimplicit/tests/conftest.py`

```python
from torchimplicit.implicit_cube import implicit_cube_3d

def _make_3d_avoid_cube_seams(margin=0.5):
    # Avoid regions where two abs-coords are close (medial axis inside, edges on surface)
    pts = _uniform(10000, 3)
    ax, ay, az = pts[:, 0].abs(), pts[:, 1].abs(), pts[:, 2].abs()
    mask = ((ax - ay).abs() < margin) | ((ax - az).abs() < margin) | ((ay - az).abs() < margin)
    pts[mask] = torch.tensor([10.0, 3.0, 1.0])
    return pts

# Add to cases_implicit_functions_3d:
pytest.param(partial(implicit_cube_3d, params=_R), _make_3d_avoid_cube_seams, id="cube")
```

## Verification

```bash
uv run pytest torchimplicit/tests/ -k cube -v
uv run pytest torchimplicit/tests/ -v
```
