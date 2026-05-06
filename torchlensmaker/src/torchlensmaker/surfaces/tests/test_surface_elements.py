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


import pytest
import torch
import torch.nn as nn

from torchlensmaker.kinematics.homogeneous_geometry import (
    hom_identity_2d,
    hom_identity_3d,
)
from torchlensmaker.surfaces import (
    ImplicitDisk,
    Asphere,
    Conic,
    Disk,
    SurfaceElementOutput,
    Parabola,
    Plane,
    PointSurface,
    Sphere,
    SphereByCurvature,
    SphereByRadius,
    XYPolynomial,
)
from torchlensmaker.types import BatchNDTensor, Tf

# --- ray generators: (N, dtype, device) -> (P, V) ---


def _rays_2d_at_origin(N: int, dtype: torch.dtype, device: torch.device):
    P = torch.zeros((N, 2), dtype=dtype, device=device)
    V = torch.tensor([[1.0, 0.0]], dtype=dtype, device=device).expand_as(P)
    return P, V


def _rays_3d_at_origin(N: int, dtype: torch.dtype, device: torch.device):
    P = torch.zeros((N, 3), dtype=dtype, device=device)
    V = torch.tensor([[1.0, 0.0, 0.0]], dtype=dtype, device=device).expand_as(P)
    return P, V


def _rays_2d_outside_r5(N: int, dtype: torch.dtype, device: torch.device):
    # Rays starting outside a sphere of radius 5 centered at origin
    P = torch.stack(
        (
            torch.full((N,), -10.0, dtype=dtype, device=device),
            torch.linspace(-1, 1, N, dtype=dtype, device=device),
        ),
        dim=-1,
    )
    V = torch.tensor([[1.0, 0.0]], dtype=dtype, device=device).expand_as(P)
    return P, V


def _rays_3d_outside_r5(N: int, dtype: torch.dtype, device: torch.device):
    # Rays starting outside a sphere of radius 5 centered at origin
    P = torch.stack(
        (
            torch.full((N,), -10.0, dtype=dtype, device=device),
            torch.linspace(-1, 1, N, dtype=dtype, device=device),
            torch.linspace(-1, 1, N, dtype=dtype, device=device),
        ),
        dim=-1,
    )
    V = torch.tensor([[1.0, 0.0, 0.0]], dtype=dtype, device=device).expand_as(P)
    return P, V


# --- test case lists ---


cases_2d = [
    pytest.param(
        Disk(10.0),
        _rays_2d_at_origin,
        id="disk",
    ),
    pytest.param(
        Plane(10.0),
        _rays_2d_at_origin,
        id="plane",
    ),
    pytest.param(
        SphereByCurvature(10.0, C=0.0),
        _rays_2d_at_origin,
        id="sphere_by_curvature_flat",
    ),
    pytest.param(
        SphereByCurvature(10.0, C=0.5),
        _rays_2d_at_origin,
        id="sphere_by_curvature_pos",
    ),
    pytest.param(
        SphereByCurvature(10.0, C=-0.5),
        _rays_2d_at_origin,
        id="sphere_by_curvature_neg",
    ),
    pytest.param(
        SphereByRadius(10, 5),
        _rays_2d_at_origin,
        id="sphere_by_radius",
    ),
    pytest.param(
        Parabola(10.0, A=0.0),
        _rays_2d_at_origin,
        id="parabola_flat",
    ),
    pytest.param(
        Parabola(10.0, A=0.5),
        _rays_2d_at_origin,
        id="parabola_pos",
    ),
    pytest.param(
        Parabola(10.0, A=-0.5),
        _rays_2d_at_origin,
        id="parabola_neg",
    ),
    pytest.param(
        Conic(10, C=0.1, K=0.1),
        _rays_2d_at_origin,
        id="conic_pos",
    ),
    pytest.param(
        Conic(10, C=-0.1, K=-0.1),
        _rays_2d_at_origin,
        id="conic_neg",
    ),
    pytest.param(
        Asphere(10, C=0.1, K=0.1, alphas=[0.1, 0.01, 0.002]),
        _rays_2d_at_origin,
        id="asphere_pos",
    ),
    pytest.param(
        Asphere(10, C=-0.1, K=-0.1, alphas=[0.1, 0.01, 0.002]),
        _rays_2d_at_origin,
        id="asphere_neg",
    ),
    pytest.param(
        Sphere(5.0),
        _rays_2d_outside_r5,
        id="sphere",
    ),
    pytest.param(
        ImplicitDisk(5.0),
        _rays_2d_at_origin,
        id="implicit_disk",
    ),
]

cases_3d = [
    pytest.param(
        Disk(10.0),
        _rays_3d_at_origin,
        id="disk",
    ),
    pytest.param(
        Plane(10.0),
        _rays_3d_at_origin,
        id="plane",
    ),
    pytest.param(
        SphereByCurvature(10, 0.05),
        _rays_3d_at_origin,
        id="sphere_by_curvature",
    ),
    pytest.param(
        SphereByRadius(10, 5),
        _rays_3d_at_origin,
        id="sphere_by_radius",
    ),
    pytest.param(
        Parabola(10.0, A=0.0),
        _rays_3d_at_origin,
        id="parabola_flat",
    ),
    pytest.param(
        Parabola(10.0, A=0.5),
        _rays_3d_at_origin,
        id="parabola_pos",
    ),
    pytest.param(
        Parabola(10.0, A=-0.5),
        _rays_3d_at_origin,
        id="parabola_neg",
    ),
    pytest.param(
        Conic(10, C=0.1, K=0.1),
        _rays_3d_at_origin,
        id="conic_pos",
    ),
    pytest.param(
        Conic(10, C=-0.1, K=-0.1),
        _rays_3d_at_origin,
        id="conic_neg",
    ),
    pytest.param(
        Asphere(10, C=0.1, K=0.1, alphas=[0.1, 0.01, 0.002]),
        _rays_3d_at_origin,
        id="asphere_pos",
    ),
    pytest.param(
        Asphere(10, C=-0.1, K=-0.1, alphas=[0.1, 0.01, 0.002]),
        _rays_3d_at_origin,
        id="asphere_neg",
    ),
    pytest.param(
        XYPolynomial(
            10, C=0.1, K=-0.1, coefficients=[[0.1, 0.2, 0.0], [0.01, 0.0, 0.01]]
        ),
        _rays_3d_at_origin,
        id="xypolynomial",
    ),
    pytest.param(
        Sphere(5.0),
        _rays_3d_outside_r5,
        id="sphere",
    ),
    pytest.param(
        ImplicitDisk(5.0),
        _rays_3d_at_origin,
        id="implicit_disk",
    ),
]


# --- check helpers ---


def check_model_eval(
    model: nn.Module, inputs: tuple[BatchNDTensor, BatchNDTensor, Tf]
) -> SurfaceElementOutput:
    "Evaluate a model forwards and run sanity checks"

    # Check the forward pass
    outputs = model(*inputs)
    assert outputs.t.isfinite().all()
    assert outputs.normals.isfinite().all()
    assert outputs.valid.isfinite().all()
    assert outputs.points_local.isfinite().all()
    assert outputs.points_global.isfinite().all()
    assert outputs.tf_surface.direct.isfinite().all()
    assert outputs.tf_surface.inverse.isfinite().all()
    assert outputs.tf_next.direct.isfinite().all()
    assert outputs.tf_next.inverse.isfinite().all()

    return outputs


def check_model_eval_and_grad(
    model: nn.Module,
    inputs: tuple[BatchNDTensor, BatchNDTensor, Tf],
    allow_none_grad: bool = False,
) -> SurfaceElementOutput:
    """
    Evaluate a model forwards and backwards and run sanity checks
    Expects at least one trainable parameter
    """

    # Check the forward pass
    outputs = model(*inputs)
    assert outputs.t.isfinite().all()
    assert outputs.normals.isfinite().all()
    assert outputs.valid.isfinite().all()
    assert outputs.points_local.isfinite().all()
    assert outputs.points_global.isfinite().all()
    assert outputs.tf_surface.direct.isfinite().all()
    assert outputs.tf_surface.inverse.isfinite().all()
    assert outputs.tf_next.direct.isfinite().all()
    assert outputs.tf_next.inverse.isfinite().all()

    # Check the backward pass
    parameters = list(model.named_parameters())
    assert len(parameters) > 0

    loss = outputs.t.pow(2).sum() + outputs.tf_next.direct.sum().pow(2)
    model.zero_grad()
    loss.backward()  # type: ignore[no-untyped-call]
    for name, param in parameters:
        print(f"grad({name}) = {param.grad}")
        assert allow_none_grad or param.grad is not None
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), (
                f"Gradient of {name} contains NaN or Inf: {param.grad}"
            )

    return outputs


def check_surface_module_2d(
    mod: nn.Module,
    trainable: bool,
    dtype: torch.dtype,
    device: torch.device,
    make_rays,
    allow_none_grad: bool = False,
) -> None:
    N = 10
    P, V = make_rays(N, dtype, device)
    tfid = hom_identity_2d(dtype, device)

    if trainable:
        outputs = check_model_eval_and_grad(mod, (P, V, tfid), allow_none_grad)
    else:
        outputs = check_model_eval(mod, (P, V, tfid))

    # Check output is sane
    assert outputs.t.shape == (N,)
    assert outputs.normals.shape == (N, 2)
    assert outputs.valid.shape == (N,)
    assert outputs.points_local.shape == (N, 2)
    assert outputs.points_global.shape == (N, 2)
    assert outputs.tf_surface.shape == (3, 3)
    assert outputs.tf_next.shape == (3, 3)

    assert outputs.t.dtype == dtype
    assert outputs.normals.dtype == dtype
    assert outputs.valid.dtype == torch.bool
    assert outputs.points_local.dtype == dtype
    assert outputs.points_global.dtype == dtype
    assert outputs.tf_surface.dtype == dtype
    assert outputs.tf_next.dtype == dtype

    assert outputs.t.device == device
    assert outputs.normals.device == device
    assert outputs.valid.device == device
    assert outputs.points_local.device == device
    assert outputs.points_global.device == device
    assert outputs.tf_surface.device == device
    assert outputs.tf_next.device == device

    # Check that surface can be cloned
    mod.clone()


def check_surface_module_3d(
    mod: nn.Module,
    trainable: bool,
    dtype: torch.dtype,
    device: torch.device,
    make_rays,
    allow_none_grad: bool = False,
) -> None:
    N = 10
    P, V = make_rays(N, dtype, device)
    tfid = hom_identity_3d(dtype, device)

    if trainable:
        outputs = check_model_eval_and_grad(mod, (P, V, tfid), allow_none_grad)
    else:
        outputs = check_model_eval(mod, (P, V, tfid))

    # Check output is sane
    assert outputs.t.shape == (N,)
    assert outputs.normals.shape == (N, 3)
    assert outputs.valid.shape == (N,)
    assert outputs.points_local.shape == (N, 3)
    assert outputs.points_global.shape == (N, 3)
    assert outputs.tf_surface.shape == (4, 4)
    assert outputs.tf_next.shape == (4, 4)

    assert outputs.t.dtype == dtype
    assert outputs.normals.dtype == dtype
    assert outputs.valid.dtype == torch.bool
    assert outputs.points_local.dtype == dtype
    assert outputs.points_global.dtype == dtype
    assert outputs.tf_surface.dtype == dtype
    assert outputs.tf_next.dtype == dtype

    assert outputs.t.device == device
    assert outputs.normals.device == device
    assert outputs.valid.device == device
    assert outputs.points_local.device == device
    assert outputs.points_global.device == device
    assert outputs.tf_surface.device == device
    assert outputs.tf_next.device == device

    # Check that surface can be cloned
    mod.clone()


# --- parametrized tests ---


@pytest.mark.parametrize("module, make_rays", cases_2d)
def test_surface_modules_2d(
    module: nn.Module, make_rays, dtype: torch.dtype, device: torch.device
) -> None:
    check_surface_module_2d(module, False, dtype, device, make_rays)
    check_surface_module_2d(module.reverse(), False, dtype, device, make_rays)


@pytest.mark.parametrize("module, make_rays", cases_3d)
def test_surface_modules_3d(
    module: nn.Module, make_rays, dtype: torch.dtype, device: torch.device
) -> None:
    check_surface_module_3d(module, False, dtype, device, make_rays)
    check_surface_module_3d(module.reverse(), False, dtype, device, make_rays)


# --- standalone tests ---


def test_point_surface_module(dtype: torch.dtype, device: torch.device) -> None:
    # Use offset rays so that no ray passes through the origin (which would produce NaN normals)
    N = 10
    P2 = torch.stack(
        (
            torch.zeros(N, dtype=dtype, device=device),
            torch.linspace(-1, 1, N, dtype=dtype, device=device),
        ),
        dim=-1,
    )
    V2 = torch.tensor([[1.0, 0.0]], dtype=dtype, device=device).expand_as(P2)
    tfid2 = hom_identity_2d(dtype, device)

    outputs2 = check_model_eval(PointSurface(), (P2, V2, tfid2))
    assert outputs2.t.shape == (N,)
    assert outputs2.normals.shape == (N, 2)
    assert outputs2.valid.shape == (N,)
    assert not outputs2.valid.any()
    assert outputs2.points_local.shape == (N, 2)
    assert outputs2.points_global.shape == (N, 2)

    P3 = torch.stack(
        (
            torch.zeros(N, dtype=dtype, device=device),
            torch.linspace(-1, 1, N, dtype=dtype, device=device),
            torch.linspace(-1, 1, N, dtype=dtype, device=device),
        ),
        dim=-1,
    )
    V3 = torch.tensor([[1.0, 0.0, 0.0]], dtype=dtype, device=device).expand_as(P3)
    tfid3 = hom_identity_3d(dtype, device)

    outputs3 = check_model_eval(PointSurface(), (P3, V3, tfid3))
    assert outputs3.t.shape == (N,)
    assert outputs3.normals.shape == (N, 3)
    assert outputs3.valid.shape == (N,)
    assert not outputs3.valid.any()
    assert outputs3.points_local.shape == (N, 3)
    assert outputs3.points_global.shape == (N, 3)

    PointSurface().clone()
    PointSurface().reverse()
