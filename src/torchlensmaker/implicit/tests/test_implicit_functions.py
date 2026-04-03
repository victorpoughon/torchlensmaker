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

from functools import partial
from typing import Callable, TypeAlias

import torch

from torchlensmaker.implicit.implicit_circle import (
    implicit_circle_2d,
    implicit_circle_3d,
)

ImplicitFunction: TypeAlias = Callable[
    [torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
]


def _fd_grad(f_fn: ImplicitFunction, points: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Compute gradient by finite difference, for testing only
    """
    N, D = points.shape
    grad = torch.zeros_like(points)
    for i in range(D):
        p_plus = points.clone()
        p_plus[:, i] += eps
        p_minus = points.clone()
        p_minus[:, i] -= eps
        grad[:, i] = (f_fn(p_plus)[0] - f_fn(p_minus)[0]) / (2 * eps)
    return grad


def _fd_hess(f_fn: ImplicitFunction, points: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Compute hessian by finite difference, for testing only
    """
    N, D = points.shape
    H = torch.zeros(N, D, D)
    for i in range(D):
        for j in range(D):
            pp = points.clone()
            pp[:, i] += eps
            pp[:, j] += eps
            pm = points.clone()
            pm[:, i] += eps
            pm[:, j] -= eps
            mp = points.clone()
            mp[:, i] -= eps
            mp[:, j] += eps
            mm = points.clone()
            mm[:, i] -= eps
            mm[:, j] -= eps
            H[:, i, j] = (f_fn(pp)[0] - f_fn(pm)[0] - f_fn(mp)[0] + f_fn(mm)[0]) / (
                4 * eps**2
            )
    return H


def check_implicit_finite_difference(
    implicit_function: ImplicitFunction,
    points: torch.Tensor,
):
    """
    Compare implicit function gradient and hessian with finite differences
    """

    dtype = torch.get_default_dtype()
    eps_grad = {
        torch.float32: 3e-3,
        torch.float64: 1e-5,
    }[dtype]
    eps_hess = {
        torch.float32: 3e-2,
        torch.float64: 1e-4,
    }[dtype]

    F, grad, hess = implicit_function(points)
    grad3_fd = _fd_grad(implicit_function, points, eps_grad)
    hess3_fd = _fd_hess(implicit_function, points, eps_hess)

    grad_error = (grad - grad3_fd).abs()
    hess_error = (hess - hess3_fd).abs()

    grad_error_min = grad_error.min()
    grad_error_mean = grad_error.mean()
    grad_error_max = grad_error.max()

    hess_error_min = hess_error.min()
    hess_error_mean = hess_error.mean()
    hess_error_max = hess_error.max()

    print(
        f"Gradient error (min/mean/max) : {grad_error_min:.2e} | {grad_error_mean:.2e} | {grad_error_max:.2e}"
    )
    print(
        f"Hessian error (min/mean/max) : {hess_error_min:.2e} | {hess_error_mean:.2e} | {hess_error_max:.2e}"
    )

    # note: we test on mean() and not max()
    # because in float32 we can have some wild outliers

    if dtype == torch.float32:
        assert grad_error_mean <= 1e-2
        assert hess_error_mean <= 1e-1
    elif dtype == torch.float64:
        assert grad_error_mean <= 1e-6
        assert hess_error_mean <= 1e-5
    else:
        raise RuntimeError(f"Unknown default dtype for test {dtype}")


def test_fd_circle_2d():
    points_2d = torch.distributions.uniform.Uniform(-20.0, 20.0).sample((10000, 2))
    points_2d[:, 1] = points_2d[:, 1].clamp(min=0.2)  # avoid r = 0

    f_2d = partial(implicit_circle_2d, R=1.5)
    check_implicit_finite_difference(f_2d, points_2d)


def test_fd_circle_3d():
    points_3d = torch.distributions.uniform.Uniform(-20.0, 20.0).sample((10000, 3))
    points_3d[:, 1:] = points_3d[:, 1:].clamp(min=0.2)  # avoid r = 0

    f_3d = partial(implicit_circle_3d, R=1.5)
    check_implicit_finite_difference(f_3d, points_3d)
