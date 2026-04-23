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

from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence, TypeAlias

import torch
from jaxtyping import Float

from torchlensmaker.core.tensor_manip import bbroad
from torchlensmaker.implicit import ImplicitFunction, ImplicitResult
from torchlensmaker.types import (
    Batch2DTensor,
    Batch3DTensor,
    BatchNDTensor,
    ScalarTensor,
)


@dataclass
class SagResult:
    val: torch.Tensor
    grad: torch.Tensor | None = field(default=None)
    hess: torch.Tensor | None = field(default=None)


class SagFunction(Protocol):
    def __call__(self, points: BatchNDTensor, *, order: int) -> SagResult: ...


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


def spherical_sag_2d(r: BatchNDTensor, C: ScalarTensor, *, order: int) -> SagResult:
    "Spherical sag in 2D, parameterized by curvature"

    r = r.squeeze(-1)
    r2 = torch.pow(r, 2)
    C2 = torch.pow(C, 2)
    g = safe_div(C * r2, 1 + safe_sqrt(1 - r2 * C2))

    g_grad = None
    if order >= 1:
        g_grad = safe_div(C * r, safe_sqrt(1 - r2 * C2)).unsqueeze(-1)

    g_hess = None
    if order >= 2:
        s = 1 - r2 * C2
        ss = safe_sqrt(s)
        g_hess = safe_div(C * (s + C2 * r2), s * ss).unsqueeze(-1).unsqueeze(-1)

    return SagResult(g, g_grad, g_hess)


def spherical_sag_3d(yz: BatchNDTensor, C: ScalarTensor, *, order: int) -> SagResult:
    "Spherical sag in 3D, parameterized by curvature"

    y, z = yz.unbind(-1)
    r2 = torch.pow(y, 2) + torch.pow(z, 2)
    C2 = torch.pow(C, 2)
    G = safe_div(C * r2, 1 + safe_sqrt(1 - r2 * C2))

    G_grad = None
    if order >= 1:
        denom = safe_sqrt(1 - r2 * C2)
        G_grad = torch.stack((safe_div(y * C, denom), safe_div(z * C, denom)), dim=-1)

    G_hess = None
    if order >= 2:
        s = 1 - r2 * C2
        ss = safe_sqrt(s)
        denom = s * ss
        Gyy = safe_div(C * (s + C2 * y**2), denom)
        Gyz = safe_div(C * C2 * y * z, denom)
        Gzz = safe_div(C * (s + C2 * z**2), denom)
        G_hess = torch.stack(
            [torch.stack([Gyy, Gyz], dim=-1), torch.stack([Gyz, Gzz], dim=-1)],
            dim=-2,
        )

    return SagResult(G, G_grad, G_hess)


def parabolic_sag_2d(r: BatchNDTensor, A: ScalarTensor, *, order: int) -> SagResult:
    "Parabolic sag in 2D"

    r = r.squeeze(-1)
    g = torch.mul(A, torch.pow(r, 2))

    g_grad = None
    if order >= 1:
        g_grad = (2.0 * A * r).unsqueeze(-1)

    g_hess = None
    if order >= 2:
        g_hess = (2.0 * A * torch.ones_like(r)).unsqueeze(-1).unsqueeze(-1)

    return SagResult(g, g_grad, g_hess)


def parabolic_sag_3d(yz: BatchNDTensor, A: ScalarTensor, *, order: int) -> SagResult:
    "Parabolic sag in 3D"

    y, z = yz.unbind(-1)
    G = torch.mul(A, (y**2 + z**2))

    G_grad = None
    if order >= 1:
        G_grad = torch.stack((2 * A * y, 2 * A * z), dim=-1)

    G_hess = None
    if order >= 2:
        diag = 2.0 * A * torch.ones_like(y)
        zeros = torch.zeros_like(y)
        G_hess = torch.stack(
            [torch.stack([diag, zeros], dim=-1), torch.stack([zeros, diag], dim=-1)],
            dim=-2,
        )

    return SagResult(G, G_grad, G_hess)


def conical_sag_2d(
    r: BatchNDTensor, C: ScalarTensor, K: ScalarTensor, *, order: int
) -> SagResult:
    "Conical sag in 2D"

    r = r.squeeze(-1)
    r2 = torch.pow(r, 2)
    C2 = torch.pow(C, 2)
    g = safe_div(C * r2, 1 + safe_sqrt(1 - (1 + K) * r2 * C2))

    g_grad = None
    if order >= 1:
        g_grad = safe_div(C * r, safe_sqrt(1 - (1 + K) * r2 * C2)).unsqueeze(-1)

    g_hess = None
    if order >= 2:
        s = 1 - (1 + K) * r2 * C2
        ss = safe_sqrt(s)
        g_hess = safe_div(C, s * ss).unsqueeze(-1).unsqueeze(-1)

    return SagResult(g, g_grad, g_hess)


def conical_sag_3d(
    yz: BatchNDTensor, C: ScalarTensor, K: ScalarTensor, *, order: int
) -> SagResult:
    "Conical sag in 3D"

    y, z = yz.unbind(-1)
    r2 = y**2 + z**2
    C2 = torch.pow(C, 2)
    G = safe_div(C * r2, 1 + safe_sqrt(1 - (1 + K) * r2 * C2))

    G_grad = None
    if order >= 1:
        denom = safe_sqrt(1 - (1 + K) * r2 * C2)
        G_grad = torch.stack((safe_div(C * y, denom), safe_div(C * z, denom)), dim=-1)

    G_hess = None
    if order >= 2:
        s = 1 - (1 + K) * r2 * C2
        ss = safe_sqrt(s)
        denom = s * ss
        Gyy = safe_div(C * (s + C2 * y**2 * (1 + K)), denom)
        Gyz = safe_div(C * C2 * y * z * (1 + K), denom)
        Gzz = safe_div(C * (s + C2 * z**2 * (1 + K)), denom)
        G_hess = torch.stack(
            [torch.stack([Gyy, Gyz], dim=-1), torch.stack([Gyz, Gzz], dim=-1)],
            dim=-2,
        )

    return SagResult(G, G_grad, G_hess)


def aspheric_sag_2d(
    r: BatchNDTensor, coefficients: Float[torch.Tensor, " C"], *, order: int
) -> SagResult:
    r = r.squeeze(-1)
    C = coefficients.shape[-1]  # number of coefficents
    alphas = bbroad(coefficients, r.dim())
    index = torch.arange(C, dtype=r.dtype, device=r.device)
    i = bbroad(index, r.dim())

    g = torch.sum(alphas * torch.pow(r, 4 + 2 * i), dim=0)

    g_grad = None
    if order >= 1:
        g_grad = torch.sum(
            alphas * (4 + 2 * i) * torch.pow(r, 3 + 2 * i), dim=0
        ).unsqueeze(-1)

    g_hess = None
    if order >= 2:
        g_hess = (
            torch.sum(
                alphas * (4 + 2 * i) * (3 + 2 * i) * torch.pow(r, 2 + 2 * i), dim=0
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

    return SagResult(g, g_grad, g_hess)


def aspheric_sag_3d(
    yz: BatchNDTensor, coefficients: Float[torch.Tensor, " C"], *, order: int
) -> SagResult:
    y, z = yz.unbind(-1)
    r2 = y**2 + z**2
    C = coefficients.shape[-1]  # number of coefficents
    alphas = bbroad(coefficients, r2.dim())
    index = torch.arange(C, dtype=r2.dtype, device=r2.device)
    i = bbroad(index, r2.dim())

    G = torch.sum(alphas * torch.pow(r2, 2 + i), dim=0)

    G_grad = None
    if order >= 1:
        coeffs_term = torch.sum(alphas * (4 + 2 * i) * torch.pow(r2, 1 + i), dim=0)
        G_grad = torch.stack((y * coeffs_term, z * coeffs_term), dim=-1)

    G_hess = None
    if order >= 2:
        T = torch.sum(alphas * (4 + 2 * i) * torch.pow(r2, 1 + i), dim=0)
        Tprime = torch.sum(alphas * (4 + 2 * i) * (1 + i) * torch.pow(r2, i), dim=0)
        Gyy = T + 2 * y**2 * Tprime
        Gyz = 2 * y * z * Tprime
        Gzz = T + 2 * z**2 * Tprime
        G_hess = torch.stack(
            [torch.stack([Gyy, Gyz], dim=-1), torch.stack([Gyz, Gzz], dim=-1)],
            dim=-2,
        )

    return SagResult(G, G_grad, G_hess)


def xypolynomial_sag_3d(
    yz: BatchNDTensor, coefficients: Float[torch.Tensor, "P Q"], *, order: int
) -> SagResult:
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

    G_grad = None
    if order >= 1:
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

    G_hess = None
    if order >= 2:
        pd2index = torch.arange(2, P, dtype=y.dtype, device=z.device)
        qd2index = torch.arange(2, Q, dtype=y.dtype, device=z.device)
        pd2, qd2 = bbroad(pd2index, y.dim()), bbroad(qd2index, y.dim())

        # Gyy
        Cpd2 = C[2:, :]
        ypd2 = torch.pow(y, pd2 - 2).unsqueeze(1)
        inner_yy = pd2.unsqueeze(1) * (pd2 - 1).unsqueeze(1) * Cpd2 * ypd2 * zq
        Gyy = inner_yy.sum(dim=0).sum(dim=0)

        # Gyz
        Cpq = C[1:, 1:]
        ypd = torch.pow(y, pd - 1).unsqueeze(1)
        zqd = torch.pow(z, qd - 1).unsqueeze(0)
        inner_yz = pd.unsqueeze(1) * qd.unsqueeze(0) * Cpq * ypd * zqd
        Gyz = inner_yz.sum(dim=0).sum(dim=0)

        # Gzz
        Cqd2 = C[:, 2:]
        zqd2 = torch.pow(z, qd2 - 2).unsqueeze(0)
        inner_zz = qd2.unsqueeze(0) * (qd2 - 1).unsqueeze(0) * Cqd2 * yp * zqd2
        Gzz = inner_zz.sum(dim=0).sum(dim=0)

        G_hess = torch.stack(
            [torch.stack([Gyy, Gyz], dim=-1), torch.stack([Gyz, Gzz], dim=-1)],
            dim=-2,
        )

    return SagResult(G, G_grad, G_hess)


def sag_sum_2d(
    r: BatchNDTensor,
    sags: Sequence[SagFunction],
    *,
    order: int,
) -> SagResult:
    # Call the sag function of each term
    results = [sag(r, order=order) for sag in sags]

    # Sum the results
    g_sum = torch.sum(torch.stack([res.val for res in results], dim=0), dim=0)

    g_grad = None
    if order >= 1:
        assert all([res.grad is not None for res in results])
        g_grad = torch.sum(torch.stack([res.grad for res in results], dim=0), dim=0)  # type: ignore

    g_hess = None
    if order >= 2:
        assert all([res.hess is not None for res in results])
        g_hess = torch.sum(torch.stack([res.hess for res in results], dim=0), dim=0)  # type: ignore
    return SagResult(g_sum, g_grad, g_hess)


def sag_sum_3d(
    yz: BatchNDTensor,
    sags: Sequence[SagFunction],
    *,
    order: int,
) -> SagResult:
    # Call the sag function of each term
    results = [sag(yz, order=order) for sag in sags]

    # Sum the results
    g_sum = torch.sum(torch.stack([res.val for res in results], dim=0), dim=0)

    g_grad = None
    if order >= 1:
        assert all([res.grad is not None for res in results])
        g_grad = torch.sum(torch.stack([res.grad for res in results], dim=0), dim=0)  # type: ignore

    g_hess = None
    if order >= 2:
        assert all([res.hess is not None for res in results])
        g_hess = torch.sum(torch.stack([res.hess for res in results], dim=0), dim=0)  # type: ignore
    return SagResult(g_sum, g_grad, g_hess)
