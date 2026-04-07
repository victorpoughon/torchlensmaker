# This file is part of Torch Lens Maker
# Copyright (C) 2024-present Victor Poughon
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from typing import Callable, Sequence, TypeAlias

import torch
from jaxtyping import Float

from torchlensmaker.core.tensor_manip import bbroad
from torchlensmaker.implicit import ImplicitFunction, ImplicitResult
from torchlensmaker.types import (
    Batch2DTensor,
    Batch3DTensor,
    BatchNDTensor,
    BatchTensor,
    ScalarTensor,
)

SagFunction: TypeAlias = Callable[[BatchNDTensor], tuple[BatchTensor, BatchNDTensor]]
"""
A sag function models a surface as a deviation from a plane. In 2D that 'plane'
is the abstract meridional axis, in 3D it's the YZ plane.

SagFunction :: points -> g(points), g_grad(points)

In 2D:
    points: tensor of shape (..., 1)
    g(points): tensor of shape (...)
    g_grad(points): tensor of shape (..., 1)

In 3D:
    points: tensor of shape (..., 2)
    g(points): tensor of shape (...)
    g_grad(points): tensor of shape (..., 2)
"""

LiftFunction: TypeAlias = Callable[
    [SagFunction, ScalarTensor, ScalarTensor],
    ImplicitFunction,
]
"""
A lift function is a transformation that turns a sag function into an implicit
function. It is also given two extra arguments:

    nf: the normalization factor
    tau: the radius of the domain of the sag function
"""


def sag_to_implicit_2d_raw(
    sag: SagFunction, nf: ScalarTensor, tau: ScalarTensor
) -> ImplicitFunction:
    """
    Wrap a 2D sag function into an implicit function

    Args:
        sag: the sag function
        nf: normalization factor, typically either 1 (no normalization) or tau (normalization enabled)

    Returns:
        An implicit function representing the surface defined by the sag function
    """

    def implicit(points: Batch2DTensor, *, order: int) -> ImplicitResult:
        x = points[..., 0]
        g, g_grad = sag(points[..., 1:] / nf)
        f = nf * g - x
        f_grad = torch.stack((-torch.ones_like(x), g_grad.squeeze(-1)), dim=-1)
        return ImplicitResult(f, f_grad, None)

    return implicit


def sag_to_implicit_2d_euclid(
    sag: SagFunction, nf: ScalarTensor, tau: ScalarTensor
) -> ImplicitFunction:
    """
    Wrap a 2D sag function into an implicit function
    Enforces lens diameter bounds with the euclian distance

    Args:
        sag: the sag function
        nf: normalization factor, typically either 1 (no normalization) or half lens diameter (normalization enabled)
        tau: domain of sag function is [-tau, tau]

    Returns:
        An implicit function representing the surface defined by the sag function
    """

    def implicit(points: Batch2DTensor, *, order: int) -> ImplicitResult:
        # inner part
        x, r = points.unbind(-1)
        g, g_grad = sag(r / nf)
        f_inner = nf * g - x
        f_grad_inner = torch.stack((-torch.ones_like(x), g_grad.squeeze(-1)), dim=-1)

        # outer part
        r_abs = torch.abs(r)
        P = torch.stack((x, r_abs), dim=-1)  # (..., 2)
        A = torch.stack((nf * sag(tau / nf)[0], tau), dim=-1)  # (2)
        f_outer = torch.linalg.vector_norm(P - A, dim=-1)
        f_grad_outer = (P - A) / f_outer.unsqueeze(-1)

        mask = r_abs <= tau
        f = torch.where(mask, f_inner, f_outer)
        f_grad = torch.where(mask.unsqueeze(-1), f_grad_inner, f_grad_outer)

        return ImplicitResult(f, f_grad, None)

    return implicit


def sag_to_implicit_3d_raw(
    sag: SagFunction, nf: ScalarTensor, tau: ScalarTensor
) -> ImplicitFunction:
    """
    Wrap a 3D sag function into an implicit function

    Args:
        sag: the sag function
        nf: normalization factor, typically either 1 (no normalization) or tau (normalization enabled)

    Returns:
        An implicit function representing the surface defined by the sag function
    """

    def implicit(points: Batch3DTensor, *, order: int) -> ImplicitResult:
        x = points[..., 0]
        g, g_grad = sag(points[..., 1:] / nf)
        f = nf * g - x
        grad_x = -torch.ones_like(x)
        grad_y, grad_z = g_grad.unbind(-1)
        f_grad = torch.stack((grad_x, grad_y, grad_z), dim=-1)

        return ImplicitResult(f, f_grad, None)

    return implicit


def safe_sqrt(radicand: torch.Tensor) -> torch.Tensor:
    """
    Gradient safe version of torch.sqrt() that returns 0 where radicand <= 0
    """
    ok = radicand > 0
    safe = torch.zeros_like(radicand)
    return torch.sqrt(torch.where(ok, radicand, safe))


def safe_div(dividend: torch.Tensor, divisor: torch.Tensor) -> torch.Tensor:
    """
    Gradient safe version of torch.div() that returns dividend where divisor == 0
    """

    ok = divisor != torch.zeros((), dtype=divisor.dtype, device=divisor.device)
    safe = torch.ones_like(divisor)
    return torch.div(dividend, torch.where(ok, divisor, safe))


def spherical_sag_2d(
    r: BatchNDTensor, C: ScalarTensor
) -> tuple[BatchTensor, BatchNDTensor]:
    "Spherical sag in 2D, parameterized by curvature"

    r = r.squeeze(-1)
    r2 = torch.pow(r, 2)
    C2 = torch.pow(C, 2)
    g = safe_div(C * r2, 1 + safe_sqrt(1 - r2 * C2))
    g_grad = safe_div(C * r, safe_sqrt(1 - torch.pow(r, 2) * C2))

    return g, g_grad.unsqueeze(-1)


def spherical_sag_3d(
    yz: BatchNDTensor, C: ScalarTensor
) -> tuple[BatchTensor, Batch2DTensor]:
    "Spherical sag in 3D, parameterized by curvature"

    y, z = yz.unbind(-1)
    r2 = torch.pow(y, 2) + torch.pow(z, 2)
    C2 = torch.pow(C, 2)
    G = safe_div(C * r2, 1 + safe_sqrt(1 - r2 * C2))
    denom = safe_sqrt(1 - r2 * C2)
    G_grad = torch.stack((safe_div(y * C, denom), safe_div(z * C, denom)), dim=-1)

    return G, G_grad


def parabolic_sag_2d(
    r: BatchNDTensor, A: ScalarTensor
) -> tuple[BatchTensor, BatchNDTensor]:
    "Parabolic sag in 2D"

    r = r.squeeze(-1)
    g = torch.mul(A, torch.pow(r, 2))
    g_grad = 2.0 * A * r
    return g, g_grad.unsqueeze(-1)


def parabolic_sag_3d(
    yz: BatchNDTensor, A: ScalarTensor
) -> tuple[BatchTensor, Batch2DTensor]:
    "Parabolic sag in 3D"

    y, z = yz.unbind(-1)
    G = torch.mul(A, (y**2 + z**2))
    G_grad = torch.stack((2 * A * y, 2 * A * z), dim=-1)
    return G, G_grad


def conical_sag_2d(
    r: BatchNDTensor, C: ScalarTensor, K: ScalarTensor
) -> tuple[BatchTensor, BatchNDTensor]:
    "Conical sag in 2D"

    r = r.squeeze(-1)
    r2 = torch.pow(r, 2)
    C2 = torch.pow(C, 2)
    g = safe_div(C * r2, 1 + safe_sqrt(1 - (1 + K) * r2 * C2))
    g_grad = safe_div(C * r, safe_sqrt(1 - (1 + K) * r2 * C2))

    return g, g_grad.unsqueeze(-1)


def conical_sag_3d(
    yz: BatchNDTensor, C: ScalarTensor, K: ScalarTensor
) -> tuple[BatchTensor, Batch2DTensor]:
    "Conical sag in 3D"

    y, z = yz.unbind(-1)
    r2 = y**2 + z**2
    C2 = torch.pow(C, 2)
    G = safe_div(C * r2, 1 + safe_sqrt(1 - (1 + K) * r2 * C2))

    denom = safe_sqrt(1 - (1 + K) * r2 * C2)
    G_grad = torch.stack((safe_div(C * y, denom), safe_div(C * z, denom)), dim=-1)

    return G, G_grad


def aspheric_sag_2d(
    r: BatchNDTensor, coefficients: Float[torch.Tensor, " C"]
) -> tuple[BatchTensor, BatchNDTensor]:
    r = r.squeeze(-1)
    C = coefficients.shape[-1]  # number of coefficents
    alphas = bbroad(coefficients, r.dim())
    index = torch.arange(C, dtype=r.dtype, device=r.device)
    i = bbroad(index, r.dim())

    g = torch.sum(alphas * torch.pow(r, 4 + 2 * i), dim=0)
    g_grad = torch.sum(alphas * (4 + 2 * i) * torch.pow(r, 3 + 2 * i), dim=0)

    return g, g_grad.unsqueeze(-1)


def aspheric_sag_3d(
    yz: BatchNDTensor, coefficients: Float[torch.Tensor, " C"]
) -> tuple[BatchTensor, Batch2DTensor]:
    y, z = yz.unbind(-1)
    r2 = y**2 + z**2
    C = coefficients.shape[-1]  # number of coefficents
    alphas = bbroad(coefficients, r2.dim())
    index = torch.arange(C, dtype=r2.dtype, device=r2.device)
    i = bbroad(index, r2.dim())

    G = torch.sum(alphas * torch.pow(r2, 2 + i), dim=0)
    coeffs_term = torch.sum(alphas * (4 + 2 * i) * torch.pow(r2, 1 + i), dim=0)
    G_grad = torch.stack((y * coeffs_term, z * coeffs_term), dim=-1)

    return G, G_grad


def xypolynomial_sag_3d(
    yz: BatchNDTensor, coefficients: Float[torch.Tensor, "P Q"]
) -> tuple[BatchTensor, Batch2DTensor]:
    r"""
    Sag function for the XY Polynomial model in 3D

    $$
    G(y,z) = \sum C_{p,q} y^p z^p
    $$
    """

    y, z = yz.unbind(-1)
    assert coefficients.dim() == 2
    assert y.shape == z.shape
    P, Q = coefficients.shape
    C = bbroad(coefficients, y.dim())

    # We need four different indexing tensors:
    # 0 to p, 1 to p, 0 to q, 1 to q
    # and each need to be broadcastable with y and z
    pindex = torch.arange(P, dtype=y.dtype, device=z.device)
    qindex = torch.arange(Q, dtype=y.dtype, device=z.device)
    pdindex = torch.arange(1, P, dtype=y.dtype, device=z.device)
    qdindex = torch.arange(1, Q, dtype=y.dtype, device=z.device)

    p, q = bbroad(pindex, y.dim()), bbroad(qindex, y.dim())
    pd, qd = bbroad(pdindex, y.dim()), bbroad(qdindex, y.dim())

    yp = torch.pow(y, p).unsqueeze(1)
    zq = torch.pow(z, q).unsqueeze(0)
    xy = yp * zq

    G = torch.sum(torch.sum(C * xy, dim=0), dim=0)

    # Slices of C that don't contain
    # the first index droped by differentiation
    Cpd = C[1:, :]
    Cqd = C[:, 1:]

    ypd = torch.pow(y, pd - 1).unsqueeze(1)
    innery = pd.unsqueeze(1) * Cpd * ypd * zq

    zqd = torch.pow(z, qd - 1).unsqueeze(0)
    innerz = qd.unsqueeze(0) * Cqd * yp * zqd

    G_grad = torch.stack(
        (
            innery.sum(dim=0).sum(dim=0),
            innerz.sum(dim=0).sum(dim=0),
        ),
        dim=-1,
    )

    return G, G_grad


def sag_sum_2d(
    r: BatchNDTensor,
    sags: Sequence[SagFunction],
) -> tuple[BatchTensor, BatchNDTensor]:
    # Call the sag function of each term
    results = [sag(r) for sag in sags]

    # Sum the results
    g_sum = torch.sum(torch.stack([g for (g, _) in results], dim=0), dim=0)
    g_grad_sum = torch.sum(
        torch.stack([g_grad for (_, g_grad) in results], dim=0), dim=0
    )
    return g_sum, g_grad_sum


def sag_sum_3d(
    yz: BatchNDTensor,
    sags: Sequence[SagFunction],
) -> tuple[BatchTensor, BatchNDTensor]:
    # Call the sag function of each term
    results = [sag(yz) for sag in sags]

    # Sum the results
    g_sum = torch.sum(torch.stack([g for (g, _) in results], dim=0), dim=0)
    g_grad_sum = torch.sum(
        torch.stack([g_grad for (_, g_grad) in results], dim=0), dim=0
    )
    return g_sum, g_grad_sum
