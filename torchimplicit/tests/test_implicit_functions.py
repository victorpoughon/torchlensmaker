from functools import partial

import pytest
import torch
from conftest import cases_2d, cases_3d

from torchimplicit.types import ImplicitFunction


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


@pytest.mark.parametrize("implicit_fn, make_points", cases_2d)
def test_implicit_2d(implicit_fn, make_points):
    points = make_points()
    check_implicit_finite_difference(implicit_fn, points)
    check_shape_dtype_device_correctness(implicit_fn, points)


@pytest.mark.parametrize("implicit_fn, make_points", cases_3d)
def test_implicit_3d(implicit_fn, make_points):
    points = make_points()
    check_implicit_finite_difference(implicit_fn, points)
    check_shape_dtype_device_correctness(implicit_fn, points)
