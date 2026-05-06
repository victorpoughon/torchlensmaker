import torch

from torchimplicit.domain import total_domain
from torchimplicit.registry import example_scalar, register_implicit_function
from torchimplicit.types import ImplicitFunction, ImplicitResult


def implicit_yzcircle_2d(
    points: torch.Tensor,
    params: torch.Tensor,
    *,
    order: int,
) -> ImplicitResult:
    """
    Implicit circle in 2D.

    Note that this is really the projection of a 3D circle into an abstract 2D meridional plane.
    So in practice it is actually the "surface" defined by the two points (0, R) and (0, -R).

        F(x, r) = sqrt((|r| - R)^2 + x^2)

    Params:
        R (scalar): circle radius

    Args:
        points (Tensor): input points, shape (..., 2) where last dimension is (x, r)
        params (Tensor): tensor of parameters
        order: differentiation order

    Returns: ImplicitResult object

    Note: singular at r = 0
    """

    assert params.shape == (1,), (
        f"yzcircle_2d expects parameters of shape (1,), got {params.shape}"
    )
    R = params[0]

    x = points[..., 0]
    r = points[..., 1]

    # --- shared intermediates ---
    sgn_r = torch.sign(r)
    d = r.abs() - R  # |r| - R
    s = d**2 + x**2
    F = s.sqrt()  # (...,)
    inv_F = 1.0 / F  # 1 / sqrt(s)

    # --- gradient ---
    grad = None
    if order >= 1:
        grad_x = x * inv_F
        grad_r = sgn_r * d * inv_F
        grad = torch.stack([grad_x, grad_r], dim=-1)  # (..., 2)

    # --- hessian ---
    hess = None
    if order >= 2:
        s32 = s * F  # s^(3/2)
        A = 1.0 / s32  # 1 / s^(3/2)
        H_xx = (s - x**2) * A
        H_xr = -x * sgn_r * d * A  # = H_rx
        H_rr = (s - d**2) * A
        hess = torch.stack(
            [
                torch.stack([H_xx, H_xr], dim=-1),
                torch.stack([H_xr, H_rr], dim=-1),
            ],
            dim=-1,
        )  # (..., 2, 2)

    return ImplicitResult(F, grad, hess)


def implicit_yzcircle_3d(
    points: torch.Tensor,
    params: torch.Tensor,
    *,
    order: int,
) -> ImplicitResult:
    """
    Implicit circle in 3D.

    The circle lies in the YZ plane, is centered at the origin and has radius R.

        F(x, y, z) = sqrt((sqrt(y^2 + z^2) - R)^2 + x^2)

    Params:
        R (scalar): circle radius

    Args:
        points (Tensor): input points, shape (..., 3) where last dimension is (x, y, z)
        params (Tensor): tensor of parameters
        order: differentiation order

    Returns: ImplicitResult object

    Note: singular on the x-axis (y = z = 0)
    """

    assert params.shape == (1,), (
        f"yzcircle_3d expects parameters of shape (1,), got {params.shape}"
    )
    R = params[0]

    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]

    # --- shared intermediates ---
    r = (y**2 + z**2).sqrt()
    d = r - R  # r - R
    s = d**2 + x**2
    F = s.sqrt()  # (...,)
    inv_F = 1.0 / F  # 1 / sqrt(s)
    inv_r = 1.0 / r  # 1 / r

    # --- gradient ---
    grad = None
    if order >= 1:
        grad_x = x * inv_F
        grad_y = y * d * inv_r * inv_F
        grad_z = z * d * inv_r * inv_F
        grad = torch.stack([grad_x, grad_y, grad_z], dim=-1)  # (..., 3)

    # --- hessian ---
    hess = None
    if order >= 2:
        s32 = s * F  # s^(3/2)
        r2 = r**2
        r3 = r * r2
        d2 = d**2
        A = 1.0 / s32  # 1 / s^(3/2)
        B = d * inv_r * A  # (r-R) / (r s^(3/2))
        C = (R * s - r * d2) / (r3 * s32)  # [Rs - r(r-R)²] / (r³ s^(3/2))
        diag_num = d * r2 * s  # (r-R) r² s
        diag_fac = R * s - r * d2  # R s - r (r-R)²

        H_xx = (s - x**2) * A
        H_yy = (diag_num + y**2 * diag_fac) / (r3 * s32)
        H_zz = (diag_num + z**2 * diag_fac) / (r3 * s32)
        H_xy = -x * y * B
        H_xz = -x * z * B
        H_yz = y * z * C

        hess = torch.stack(
            [
                torch.stack([H_xx, H_xy, H_xz], dim=-1),
                torch.stack([H_xy, H_yy, H_yz], dim=-1),
                torch.stack([H_xz, H_yz, H_zz], dim=-1),
            ],
            dim=-1,
        )  # (..., 3, 3)

    return ImplicitResult(F, grad, hess)


yzcircle_2d = ImplicitFunction(
    name="yzcircle_2d",
    dim=2,
    func=implicit_yzcircle_2d,
    n_params=1,
    param_names=("R",),
    example_params=example_scalar(5.0),
    domain=total_domain,
)

register_implicit_function(yzcircle_2d)


yzcircle_3d = ImplicitFunction(
    name="yzcircle_3d",
    dim=3,
    func=implicit_yzcircle_3d,
    n_params=1,
    param_names=("R",),
    example_params=example_scalar(5.0),
    domain=total_domain,
)

register_implicit_function(yzcircle_3d)
