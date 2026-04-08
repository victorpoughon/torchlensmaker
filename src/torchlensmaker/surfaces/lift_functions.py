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

from typing import Callable, TypeAlias

import torch
from jaxtyping import Float
from sympy.polys.galoistools import gf_value

from torchlensmaker.implicit import ImplicitFunction, ImplicitResult
from torchlensmaker.surfaces.sag_functions import SagFunction, SagResult
from torchlensmaker.types import (
    Batch2DTensor,
    Batch3DTensor,
    BatchNDTensor,
    ScalarTensor,
)

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
        g = sag(points[..., 1:] / nf, order=order)
        f = nf * g.val - x
        assert f.shape == points.shape[:-1]

        f_grad = None
        if order >= 1:
            assert g.grad is not None
            f_grad = torch.stack((-torch.ones_like(x), g.grad.squeeze(-1)), dim=-1)

        f_hess = None
        if order >= 2:
            assert g.hess is not None
            zeros = torch.zeros_like(x)
            H_rr = g.hess.squeeze(-1).squeeze(-1) / nf
            f_hess = torch.stack(
                [
                    torch.stack([zeros, zeros], dim=-1),
                    torch.stack([zeros, H_rr], dim=-1),
                ],
                dim=-1,
            )
            assert f_hess.shape == (*points.shape[:-1], 2, 2)

        return ImplicitResult(f, f_grad, f_hess)

    return implicit


def sag_taylor_expansion_2d(
    sag: SagFunction, nf: ScalarTensor, tau: ScalarTensor
) -> SagFunction:
    "Sag function defined by the taylor expansion of another at the lens boundary"

    bound = sag((tau / nf).unsqueeze(0), order=2)

    def extended(points: torch.Tensor, *, order: int) -> SagResult:
        assert bound.grad is not None
        assert bound.hess is not None
        r = torch.abs(points[..., 0])

        g_val = bound.val + bound.grad * (r - tau) + 1 / 2 * bound.hess * (r - tau) ** 2
        assert g_val.shape == r.shape

        g_grad = None
        if order >= 1:
            g_grad = (bound.grad + bound.hess * (r - tau)).unsqueeze(-1)
            assert g_grad.shape == (*r.shape, 1)

        g_hess = None
        if order >= 2:
            g_hess = (bound.hess.expand_as(r)).unsqueeze(-1).unsqueeze(-1)
            assert g_hess.shape == (*r.shape, 1, 1), g_hess

        return SagResult(g_val, g_grad, g_hess)

    return extended


def sag_to_implicit_2d_taylor(
    sag: SagFunction, nf: ScalarTensor, tau: ScalarTensor
) -> ImplicitFunction:
    sag_exp = sag_taylor_expansion_2d(sag, nf, tau)

    def implicit(points: Batch2DTensor, *, order: int) -> ImplicitResult:
        x = points[..., 0]
        g = sag_exp(points[..., 1:] / nf, order=order)
        f = nf * g.val - x
        assert f.shape == points.shape[:-1]

        f_grad = None
        if order >= 1:
            assert g.grad is not None
            f_grad = torch.stack((-torch.ones_like(x), g.grad.squeeze(-1)), dim=-1)
            assert f_grad.shape == (*points.shape[:-1], 2)

        f_hess = None
        if order >= 2:
            assert g.hess is not None
            zeros = torch.zeros_like(x)
            H_rr = g.hess.squeeze(-1).squeeze(-1) / nf
            f_hess = torch.stack(
                [
                    torch.stack([zeros, zeros], dim=-1),
                    torch.stack([zeros, H_rr], dim=-1),
                ],
                dim=-1,
            )
            assert f_hess.shape == (*points.shape[:-1], 2, 2)

        return ImplicitResult(f, f_grad, f_hess)

    return implicit


def sag_to_implicit_2d_taylor_squared(
    sag: SagFunction, nf: ScalarTensor, tau: ScalarTensor
) -> ImplicitFunction:
    """
    Taylor squared lift function, aka F_in

    F_in(x, r) = (x - nf * g(r / nf))^2

    where g here is the taylor expanded sag function
    """
    sag_exp = sag_taylor_expansion_2d(sag, nf, tau)

    def implicit(points: Batch2DTensor, *, order: int) -> ImplicitResult:
        x = points[..., 0]
        g = sag_exp(points[..., 1:] / nf, order=order)
        e = x - nf * g.val
        f = e**2
        assert f.shape == points.shape[:-1]

        f_grad = None
        if order >= 1:
            assert g.grad is not None
            gp = g.grad.squeeze(-1)
            f_grad = torch.stack((2 * e, (-2 * e * gp)), dim=-1)
            assert f_grad.shape == (*points.shape[:-1], 2)

        f_hess = None
        if order >= 2:
            assert g.grad is not None
            assert g.hess is not None
            gp = g.grad.squeeze(-1)
            gpp = g.hess.squeeze(-1).squeeze(-1)
            H_xx = 2 * torch.ones_like(x)
            H_rx = -2 * gp
            H_rr = 2 * gp**2 - (2 * e / nf) * gpp
            f_hess = torch.stack(
                [
                    torch.stack([H_xx, H_rx], dim=-1),
                    torch.stack([H_rx, H_rr], dim=-1),
                ],
                dim=-1,
            )
            assert f_hess.shape == (*points.shape[:-1], 2, 2)

        return ImplicitResult(f, f_grad, f_hess)

    return implicit


def sag_to_implicit_2d_euclid_squared(
    sag: SagFunction, nf: ScalarTensor, tau: ScalarTensor
) -> ImplicitFunction:
    """
    Squared euclidian distance to the lens boundary
    """
    x_a = nf * sag(tau / nf, order=0).val

    def implicit(points: Batch2DTensor, *, order: int) -> ImplicitResult:
        x, r = points.unbind(-1)
        r_abs = torch.abs(r)
        f = (x - x_a) ** 2 + (r_abs - tau) ** 2
        assert f.shape == points.shape[:-1]

        f_grad = None
        if order >= 1:
            f_grad = 2 * torch.stack(((x - x_a), safe_sign(r) * (r_abs - tau)), dim=-1)
            assert f_grad.shape == (*points.shape[:-1], 2)

        f_hess = None
        if order >= 2:
            f_hess = (
                2 * torch.eye(2, dtype=points.dtype, device=points.device)
            ).expand((*points.shape[:-1], 2, 2))
            assert f_hess.shape == (*points.shape[:-1], 2, 2)

        return ImplicitResult(f, f_grad, f_hess)

    return implicit


def sag_to_implicit_2d_squared_blend(
    sag: SagFunction,
    nf: ScalarTensor,
    tau: ScalarTensor,
) -> ImplicitFunction:
    """
    Smooth C^2 blend of the Taylor-squared lift (F_in) and the Euclidean-squared
    lift (F_out), using a quintic smooth step sigma(u) = 6u^5 - 15u^4 + 10u^3
    over the transition parameter u = (|r| - tau) / w.

    - For |r| <= tau:       F_blend = F_in  (exactly, via sigma clamped to 0)
    - For |r| >= tau + w:   F_blend = F_out (exactly, via sigma clamped to 1)
    - In between:           smooth C^2 interpolation

    The result is a C^2 scalar field on the full 2D ambient space, zero on the
    lens surface and equal to the squared distance to the rim point outside.
    """
    f_in = sag_to_implicit_2d_taylor_squared(sag, nf, tau)
    f_out = sag_to_implicit_2d_euclid_squared(sag, nf, tau)

    w = 2.5  # TODO argument

    def implicit(points: Batch2DTensor, *, order: int) -> ImplicitResult:
        r = points[..., 1]

        # Evaluate both sub-lifts at the requested order
        res_in = f_in(points, order=order)
        res_out = f_out(points, order=order)

        # Transition parameter, clamped to [0, 1]
        u = ((r.abs() - tau) / w).clamp(0.0, 1.0)

        # Smooth step and derivatives (with respect to u)
        u2 = u * u
        u3 = u2 * u
        u4 = u3 * u
        u5 = u4 * u
        alpha = 6 * u5 - 15 * u4 + 10 * u3

        # Blend value: F = F_in + alpha * (F_out - F_in)
        delta_val = res_out.val - res_in.val
        f = res_in.val + alpha * delta_val
        assert f.shape == points.shape[:-1]

        f_grad = None
        f_hess = None
        if order >= 1:
            assert res_in.grad is not None
            assert res_out.grad is not None

            sigma_p = 30 * u4 - 60 * u3 + 30 * u2  # sigma'(u)

            # Chain rule: d/dr of alpha = sigma'(u) * sgn(r) / w.
            # alpha has no dependence on x.
            s = safe_sign(r)
            alpha_r = sigma_p * s / w  # shape: batch

            # Gradient of delta = F_out - F_in
            delta_grad = res_out.grad - res_in.grad  # shape: (..., 2)

            # Blend gradient:
            #   d_x F = (1 - alpha) d_x F_in + alpha d_x F_out
            #   d_r F = (1 - alpha) d_r F_in + alpha d_r F_out + alpha_r * delta_val
            f_grad = res_in.grad + alpha.unsqueeze(-1) * delta_grad
            # Add the alpha_r * delta_val contribution to the r-component only
            extra_r = alpha_r * delta_val
            f_grad = f_grad + torch.stack((torch.zeros_like(extra_r), extra_r), dim=-1)
            assert f_grad.shape == (*points.shape[:-1], 2)

            if order >= 2:
                assert res_in.grad is not None
                assert res_out.grad is not None
                assert res_in.hess is not None
                assert res_out.hess is not None

                sigma_pp = 120 * u3 - 180 * u2 + 60 * u  # sigma''(u)

                # alpha_rr = sigma''(u) / w^2  (since (d|r|/dr)^2 = 1 and d^2|r|/dr^2 = 0
                # away from r = 0, which never occurs in the transition band)
                alpha_rr = sigma_pp / (w * w)

                # Gradient of delta (needed for the symmetrized cross term)
                delta_grad_x = delta_grad[..., 0]
                delta_grad_r = delta_grad[..., 1]

                # Hessian of delta
                delta_hess = res_out.hess - res_in.hess  # shape: (..., 2, 2)

                # Blend Hessian formula:
                #   H = H_in + alpha * H_delta
                #     + [ 0,                  alpha_r * delta_grad_x ]
                #       [ alpha_r * dgx,      2 alpha_r * dgr + alpha_rr * delta_val ]
                f_hess = res_in.hess + alpha.unsqueeze(-1).unsqueeze(-1) * delta_hess

                H_xx_extra = torch.zeros_like(alpha)
                H_xr_extra = alpha_r * delta_grad_x
                H_rr_extra = 2 * alpha_r * delta_grad_r + alpha_rr * delta_val

                extra = torch.stack(
                    [
                        torch.stack([H_xx_extra, H_xr_extra], dim=-1),
                        torch.stack([H_xr_extra, H_rr_extra], dim=-1),
                    ],
                    dim=-2,
                )
                f_hess = f_hess + extra
                assert f_hess.shape == (*points.shape[:-1], 2, 2)

        return ImplicitResult(f, f_grad, f_hess)

    return implicit


def safe_sign(x: torch.Tensor) -> torch.Tensor:
    "Like torch.sign() but equals 1 at 0"
    ones = torch.ones_like(x)
    return torch.where(x >= 0, ones, -ones)


def sag_to_implicit_2d_abs(
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
        g = sag(points[..., 1:] / nf, order=order)
        core = nf * g.val - x
        f = torch.abs(core)
        assert f.shape == points.shape[:-1]
        s = safe_sign(core)

        f_grad = None
        if order >= 1:
            assert g.grad is not None
            f_grad = torch.stack((-s, s * g.grad.squeeze(-1)), dim=-1)
            assert f_grad.shape == (*points.shape[:-1], 2)

        f_hess = None
        if order >= 2:
            assert g.hess is not None
            zeros = torch.zeros_like(x)
            H_rr = s * g.hess.squeeze(-1).squeeze(-1) / nf
            f_hess = torch.stack(
                [
                    torch.stack([zeros, zeros], dim=-1),
                    torch.stack([zeros, H_rr], dim=-1),
                ],
                dim=-1,
            )
            assert f_hess.shape == (*points.shape[:-1], 2, 2)

        return ImplicitResult(f, f_grad, f_hess)

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

    raw_implicit = sag_to_implicit_2d_abs(sag, nf, tau)

    def implicit(points: Batch2DTensor, *, order: int) -> ImplicitResult:
        # inner part: delegate to raw
        F_inner = raw_implicit(points, order=order)

        # outer part
        x, r = points.unbind(-1)
        r_abs = torch.abs(r)
        P = torch.stack((x, r_abs), dim=-1)  # (..., 2)
        A = torch.stack((nf * sag(tau / nf, order=0).val, tau), dim=-1)  # (2)
        f_outer = torch.linalg.vector_norm(P - A, dim=-1)

        mask = r_abs <= tau
        f = torch.where(mask, F_inner.val, f_outer)
        assert f.shape == points.shape[:-1]

        f_grad = None
        if order >= 1:
            assert F_inner.grad is not None
            f_grad_outer = (P - A) / f_outer.unsqueeze(-1)
            f_grad = torch.where(mask.unsqueeze(-1), F_inner.grad, f_grad_outer)
            assert f_grad.shape == (*points.shape[:-1], 2)

        f_hess = None
        if order >= 2:
            assert F_inner.hess is not None
            # outer: hessian of euclidean distance ‖P - A‖
            d = P - A  # (..., 2)
            f2 = f_outer**2
            f3 = f_outer * f2
            H_xx = (f2 - d[..., 0] ** 2) / f3
            H_xr = -d[..., 0] * d[..., 1] / f3
            H_rr = (f2 - d[..., 1] ** 2) / f3
            f_hess_outer = torch.stack(
                [
                    torch.stack([H_xx, H_xr], dim=-1),
                    torch.stack([H_xr, H_rr], dim=-1),
                ],
                dim=-1,
            )
            f_hess = torch.where(
                mask.unsqueeze(-1).unsqueeze(-1), F_inner.hess, f_hess_outer
            )
            assert f_hess.shape == (*points.shape[:-1], 2, 2)

        return ImplicitResult(f, f_grad, f_hess)

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
        g = sag(points[..., 1:] / nf, order=order)
        f = nf * g.val - x
        assert f.shape == points.shape[:-1]

        f_grad = None
        if order >= 1:
            assert g.grad is not None
            grad_x = -torch.ones_like(x)
            grad_y, grad_z = g.grad.unbind(-1)
            f_grad = torch.stack((grad_x, grad_y, grad_z), dim=-1)
            assert f_grad.shape == (*points.shape[:-1], 2)

        f_hess = None
        if order >= 2:
            assert g.hess is not None
            zeros = torch.zeros_like(x)
            Gyy = g.hess[..., 0, 0] / nf
            Gyz = g.hess[..., 0, 1] / nf
            Gzz = g.hess[..., 1, 1] / nf
            f_hess = torch.stack(
                [
                    torch.stack([zeros, zeros, zeros], dim=-1),
                    torch.stack([zeros, Gyy, Gyz], dim=-1),
                    torch.stack([zeros, Gyz, Gzz], dim=-1),
                ],
                dim=-1,
            )
            assert f_hess.shape == (*points.shape[:-1], 2, 2)

        return ImplicitResult(f, f_grad, f_hess)

    return implicit
