import torch

from torchimplicit.domain import total_domain
from torchimplicit.math import safe_div, safe_sign
from torchimplicit.registry import example_scalar, register_implicit_function
from torchimplicit.types import ImplicitFunction, ImplicitResult


def implicit_cube_3d(
    points: torch.Tensor,
    params: torch.Tensor,
    *,
    order: int,
) -> ImplicitResult:
    """
    Signed distance function for a cube centered at the origin.

        q = (|x| - a, |y| - a, |z| - a)   where a = s / 2
        F = ||max(q, 0)|| + min(max(qx, qy, qz), 0)

    F > 0 outside, F = 0 on the surface, F < 0 inside.

    Params:
        s (scalar): side length

    Args:
        points (Tensor): input points, shape (..., 3)
        params (Tensor): tensor of parameters
        order: differentiation order

    Note: not differentiable at cube surface edges/corners and interior medial axis.
    """

    assert params.shape == (1,), (
        f"cube_3d expects parameters of shape (1,), got {params.shape}"
    )
    a = params[0] / 2

    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]

    ax = x.abs()
    ay = y.abs()
    az = z.abs()

    qx = ax - a
    qy = ay - a
    qz = az - a

    qxc = qx.clamp(min=0)
    qyc = qy.clamp(min=0)
    qzc = qz.clamp(min=0)

    outer = (qxc**2 + qyc**2 + qzc**2).sqrt()
    inner = torch.maximum(torch.maximum(qx, qy), qz).clamp(max=0)

    F = outer + inner

    # Unnormalized outer-gradient components — zero inside (where all qxc=0)
    sx = safe_sign(x)
    sy = safe_sign(y)
    sz = safe_sign(z)
    gxc = sx * qxc
    gyc = sy * qyc
    gzc = sz * qzc

    # --- gradient ---
    grad = None
    if order >= 1:
        outside = outer > 0
        inv_outer = safe_div(torch.ones_like(outer), outer)

        # Inside: gradient is unit normal toward the closest face
        x_dom = (qx >= qy) & (qx >= qz)
        y_dom = (qy > qx) & (qy >= qz)
        zeros = torch.zeros_like(x)

        grad_x = torch.where(outside, gxc * inv_outer, torch.where(x_dom, sx, zeros))
        grad_y = torch.where(outside, gyc * inv_outer, torch.where(y_dom, sy, zeros))
        grad_z = torch.where(outside, gzc * inv_outer, torch.where(x_dom | y_dom, zeros, sz))

        grad = torch.stack([grad_x, grad_y, grad_z], dim=-1)

    # --- hessian ---
    hess = None
    if order >= 2:
        outside = outer > 0
        zeros = torch.zeros_like(x)
        inv_outer = safe_div(torch.ones_like(outer), outer)
        inv_outer3 = inv_outer**3

        # Outside: H[i,j] = di*delta_ij/outer - gic*gjc/outer^3
        # Inside: H = 0 (piecewise linear function)
        dx = qx.gt(0).to(x.dtype)
        dy = qy.gt(0).to(x.dtype)
        dz = qz.gt(0).to(x.dtype)

        H_xx = torch.where(outside, dx * inv_outer - gxc**2 * inv_outer3, zeros)
        H_yy = torch.where(outside, dy * inv_outer - gyc**2 * inv_outer3, zeros)
        H_zz = torch.where(outside, dz * inv_outer - gzc**2 * inv_outer3, zeros)
        H_xy = torch.where(outside, -gxc * gyc * inv_outer3, zeros)
        H_xz = torch.where(outside, -gxc * gzc * inv_outer3, zeros)
        H_yz = torch.where(outside, -gyc * gzc * inv_outer3, zeros)

        hess = torch.stack(
            [
                torch.stack([H_xx, H_xy, H_xz], dim=-1),
                torch.stack([H_xy, H_yy, H_yz], dim=-1),
                torch.stack([H_xz, H_yz, H_zz], dim=-1),
            ],
            dim=-1,
        )  # (..., 3, 3)

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
