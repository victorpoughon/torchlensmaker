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

import torch

from torchlensmaker.implicit.implicit_circle import (
    implicit_yzcircle_2d,
    implicit_yzcircle_3d,
)
from torchlensmaker.implicit.implicit_plane import (
    implicit_yaxis_2d,
    implicit_yzplane_3d,
)
from torchlensmaker.implicit.implicit_sphere import (
    implicit_sphere_2d,
    implicit_sphere_3d,
)
from torchlensmaker.implicit.types import ImplicitFunction


def _fd_grad(
    implicit_function: ImplicitFunction, points: torch.Tensor, eps: float
) -> torch.Tensor:
    """
    Compute gradient by finite difference, for testing only
    """
    f_fn = partial(implicit_function, order=0)

    N, D = points.shape
    grad = torch.zeros_like(points)
    for i in range(D):
        p_plus = points.clone()
        p_plus[:, i] += eps
        p_minus = points.clone()
        p_minus[:, i] -= eps
        grad[:, i] = (f_fn(p_plus).val - f_fn(p_minus).val) / (2 * eps)
    return grad


def _fd_hess(
    implicit_function: ImplicitFunction, points: torch.Tensor, eps: float
) -> torch.Tensor:
    """
    Compute hessian by finite difference, for testing only
    """
    f_fn = partial(implicit_function, order=0)
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
            H[:, i, j] = (f_fn(pp).val - f_fn(pm).val - f_fn(mp).val + f_fn(mm).val) / (
                4 * eps**2
            )
    return H


def check_shape_dtype_device_correctness(
    implicit_function: ImplicitFunction,
    points: torch.Tensor,
):
    dtype, device = points.dtype, points.device

    result = implicit_function(points, order=2)
    F, grad, hess = result.val, result.grad, result.hess

    assert grad is not None
    assert hess is not None

    print(f"Checking dtype={dtype}")
    assert F.dtype == dtype
    assert grad.dtype == dtype
    assert hess.dtype == dtype

    print(f"Checking device={device}")
    assert F.device == device
    assert grad.device == device
    assert hess.device == device

    print("Checking shape")
    N = points.shape[:-1]
    D = points.shape[-1]
    assert F.shape == N
    assert grad.shape == (*N, D)
    assert hess.shape == (*N, D, D)


def check_implicit_finite_difference(
    implicit_function: ImplicitFunction,
    points: torch.Tensor,
):
    """
    Compare implicit function gradient and hessian with finite differences
    """

    dtype = points.dtype
    eps_grad = {
        torch.float32: 3e-3,
        torch.float64: 1e-5,
    }[dtype]
    eps_hess = {
        torch.float32: 3e-2,
        torch.float64: 1e-4,
    }[dtype]

    result = implicit_function(points, order=2)
    F, grad, hess = result.val, result.grad, result.hess

    assert grad is not None
    assert hess is not None

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

    f_2d = partial(implicit_yzcircle_2d, R=1.5)
    check_implicit_finite_difference(f_2d, points_2d)
    check_shape_dtype_device_correctness(f_2d, points_2d)


def test_fd_circle_3d():
    points_3d = torch.distributions.uniform.Uniform(-20.0, 20.0).sample((10000, 3))
    points_3d[:, 1:] = points_3d[:, 1:].clamp(min=0.2)  # avoid r = 0

    f_3d = partial(implicit_yzcircle_3d, R=1.5)
    check_implicit_finite_difference(f_3d, points_3d)
    check_shape_dtype_device_correctness(f_3d, points_3d)


def test_fd_sphere_2d():
    points_2d = torch.distributions.uniform.Uniform(-20.0, 20.0).sample((10000, 2))
    # avoid origin where function is singular
    mask = (points_2d**2).sum(dim=-1).sqrt() < 0.2
    points_2d[mask] = torch.tensor([0.5, 0.5])

    f_2d = partial(implicit_sphere_2d, R=1.5)
    check_implicit_finite_difference(f_2d, points_2d)
    check_shape_dtype_device_correctness(f_2d, points_2d)


def test_fd_sphere_3d():
    points_3d = torch.distributions.uniform.Uniform(-20.0, 20.0).sample((10000, 3))
    # avoid origin where function is singular
    mask = (points_3d**2).sum(dim=-1).sqrt() < 0.2
    points_3d[mask] = torch.tensor([0.5, 0.5, 0.5])

    f_3d = partial(implicit_sphere_3d, R=1.5)
    check_implicit_finite_difference(f_3d, points_3d)
    check_shape_dtype_device_correctness(f_3d, points_3d)


def test_fd_yplane_2d():
    points_2d = torch.distributions.uniform.Uniform(-20.0, 20.0).sample((10000, 2))

    f_2d = partial(implicit_yaxis_2d)
    check_implicit_finite_difference(f_2d, points_2d)
    check_shape_dtype_device_correctness(f_2d, points_2d)


def test_fd_yplane_3d():
    points_3d = torch.distributions.uniform.Uniform(-20.0, 20.0).sample((10000, 3))

    f_3d = partial(implicit_yzplane_3d)
    check_implicit_finite_difference(f_3d, points_3d)
    check_shape_dtype_device_correctness(f_3d, points_3d)
