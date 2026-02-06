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


from typing import TypeAlias, Any, Sequence, Callable
from jaxtyping import Float
import torch

from torchlensmaker.core.tensor_manip import bbroad


BatchTensor: TypeAlias = Float[torch.Tensor, "..."]
Batch2DTensor: TypeAlias = Float[torch.Tensor, "... 2"]
ScalarTensor: TypeAlias = Float[torch.Tensor, ""]

# r -> g(r), g_grad(r)
SagFunction2D: TypeAlias = Callable[[BatchTensor], tuple[BatchTensor, BatchTensor]]

# y, z -> g(y, z), g_grad(y, z)
SagFunction3D = Callable[[BatchTensor], tuple[BatchTensor, Batch2DTensor]]



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

    ok = divisor != torch.zeros((), dtype=divisor.dtype)
    safe = torch.ones_like(divisor)
    return torch.div(dividend, torch.where(ok, divisor, safe))


def spherical_sag_2d(
    r: BatchTensor, C: ScalarTensor
) -> tuple[BatchTensor, BatchTensor]:
    "Spherical sag in 2D, parameterized by curvature"

    r2 = torch.pow(r, 2)
    g = safe_div(C * r2, 1 + safe_sqrt(1 - r2 * torch.pow(C, 2)))
    g_grad = safe_div(C * r, safe_sqrt(1 - torch.pow(r, 2) * torch.pow(C, 2)))

    return g, g_grad


def spherical_sag_2d_domain(C: ScalarTensor) -> Float[torch.Tensor, " 2"]:
    sign = torch.sign(C)
    return torch.stack((-sign / C, sign / C), dim=-1)


def spherical_sag_3d(
    y: BatchTensor, z: BatchTensor, C: ScalarTensor
) -> tuple[BatchTensor, Batch2DTensor]:
    "Spherical sag in 3D, parameterized by curvature"

    r2 = torch.pow(y, 2) + torch.pow(z, 2)
    G = safe_div(C * r2, 1 + safe_sqrt(1 - r2 * torch.pow(C, 2)))
    denom = safe_sqrt(1 - r2 * torch.pow(C, 2))
    G_grad = torch.stack((safe_div(y * C, denom), safe_div(z * C, denom)), dim=-1)

    return G, G_grad


def parabolic_sag_2d(
    r: BatchTensor, A: ScalarTensor
) -> tuple[BatchTensor, BatchTensor]:
    "Parabolic sag in 2D"

    g = torch.mul(A, torch.pow(r, 2))
    g_grad = 2.0 * A * r
    return g, g_grad


def parabolic_sag_3d(
    y: BatchTensor, z: BatchTensor, A: ScalarTensor
) -> tuple[BatchTensor, Batch2DTensor]:
    "Parabolic sag in 3D"

    G = torch.mul(A, (y**2 + z**2))
    G_grad = torch.stack((2 * A * y, 2 * A * z), dim=-1)
    return G, G_grad


def conical_sag_2d(
    r: BatchTensor, C: ScalarTensor, K: ScalarTensor
) -> tuple[BatchTensor, BatchTensor]:
    "Conical sag in 2D"

    r2 = torch.pow(r, 2)
    C2 = torch.pow(C, 2)
    g = torch.div(C * r2, 1 + torch.sqrt(1 - (1 + K) * r2 * C2))
    g_grad = torch.div(C * r, torch.sqrt(1 - (1 + K) * r2 * C2))

    return g, g_grad


def conical_sag_3d(
    y: BatchTensor, z: BatchTensor, C: ScalarTensor, K: ScalarTensor
) -> tuple[BatchTensor, Batch2DTensor]:
    "Conical sag in 3D"

    C2 = torch.pow(C, 2)
    r2 = y**2 + z**2
    G = torch.div(C * r2, 1 + torch.sqrt(1 - (1 + K) * r2 * C2))

    denom = torch.sqrt(1 - (1 + K) * r2 * C2)
    G_grad = torch.stack(((C * y) / denom, (C * z) / denom), dim=-1)

    return G, G_grad


def aspheric_sag_2d(
    r: BatchTensor, coefficients: Float[torch.Tensor, " C"]
) -> tuple[BatchTensor, BatchTensor]:
    C = coefficients.shape[-1]  # number of coefficents
    alphas = bbroad(coefficients, r.dim())
    index = torch.arange(C, dtype=r.dtype, device=r.device)
    i = bbroad(index, r.dim())

    g = torch.sum(alphas * torch.pow(r, 4 + 2 * i), dim=0)
    g_grad = torch.sum(alphas * (4 + 2 * i) * torch.pow(r, 3 + 2 * i), dim=0)

    return g, g_grad


def aspheric_sag_3d(
    y: BatchTensor, z: BatchTensor, coefficients: Float[torch.Tensor, " C"]
) -> tuple[BatchTensor, Batch2DTensor]:
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
    y: BatchTensor, z: BatchTensor, coefficients: Float[torch.Tensor, "P Q"]
) -> tuple[BatchTensor, Batch2DTensor]:
    r"""
    Sag function for the XY Polynomial model in 3D

    $$
    G(y,z) = \sum C_{p,q} y^p z^p
    $$
    """

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
    r: BatchTensor,
    sags: Sequence[Callable[[BatchTensor], tuple[BatchTensor, BatchTensor]]],
) -> tuple[BatchTensor, BatchTensor]:
    # Call the sag function of each term
    results = [sag(r) for sag in sags]

    # Sum the results
    g_sum = torch.sum(torch.stack([g for (g, _) in results], dim=0), dim=0)
    g_grad_sum = torch.sum(
        torch.stack([g_grad for (_, g_grad) in results], dim=0), dim=0
    )
    return g_sum, g_grad_sum


def sag_sum_3d(
    y: BatchTensor,
    z: BatchTensor,
    sags: Sequence[Callable[[BatchTensor, BatchTensor], tuple[BatchTensor, Batch2DTensor]]],
) -> tuple[BatchTensor, BatchTensor]:
    # Call the sag function of each term
    results = [sag(y, z) for sag in sags]

    # Sum the results
    g_sum = torch.sum(torch.stack([g for (g, _) in results], dim=0), dim=0)
    g_grad_sum = torch.sum(
        torch.stack([g_grad for (_, g_grad) in results], dim=0), dim=0
    )
    return g_sum, g_grad_sum
