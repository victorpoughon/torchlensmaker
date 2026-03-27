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


from itertools import chain
from pathlib import Path
from typing import Any, Dict

import onnxruntime
import pytest
import torch
import torch.nn as nn

from torchlensmaker.kinematics.homogeneous_geometry import (
    hom_identity_2d,
    hom_identity_3d,
)
from torchlensmaker.surfaces.surface_asphere import Asphere
from torchlensmaker.surfaces.surface_conic import Conic
from torchlensmaker.surfaces.surface_disk import Disk
from torchlensmaker.surfaces.surface_element import SurfaceElementOutput
from torchlensmaker.surfaces.surface_parabola import Parabola
from torchlensmaker.surfaces.surface_sphere_by_curvature import (
    SphereByCurvature,
)
from torchlensmaker.surfaces.surface_sphere_by_radius import SphereByRadius
from torchlensmaker.surfaces.surface_xypolynomial import XYPolynomial
from torchlensmaker.types import BatchNDTensor, Tf


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
    allow_none_grad: bool = False,
) -> None:
    # Check that surface model can be evaluated and differentiated
    N = 10
    P = torch.zeros((N, 2), dtype=dtype, device=device)
    V = torch.tensor([1.0, 0.0], dtype=dtype, device=device).expand_as(P)
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
    allow_none_grad: bool = False,
) -> None:
    # Check that surface model can be evaluated and differentiated
    N = 10
    P = torch.zeros((N, 3), dtype=dtype, device=device)
    V = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device).expand_as(P)
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


def test_sag_surfaces_modules_2d(dtype: torch.dtype, device: torch.device) -> None:
    surfaces_2d = [
        Disk(10.0),
        SphereByCurvature(10.0, C=0.0),
        SphereByCurvature(10.0, C=0.5),
        SphereByCurvature(10.0, C=-0.5),
        SphereByRadius(10, 5),
        Parabola(10.0, A=-0.0),
        Parabola(10.0, A=0.5),
        Parabola(10.0, A=-0.5),
        Conic(10, C=0.1, K=0.1),
        Conic(10, C=-0.1, K=-0.1),
        Asphere(10, C=0.1, K=0.1, alphas=[0.1, 0.01, 0.002]),
        Asphere(10, C=-0.1, K=-0.1, alphas=[0.1, 0.01, 0.002]),
    ]

    for module in surfaces_2d:
        check_surface_module_2d(module, False, dtype, device)
        check_surface_module_2d(module.reverse(), False, dtype, device)


def test_sag_surfaces_modules_3d(dtype: torch.dtype, device: torch.device) -> None:
    surfaces_3d = [
        Disk(10.0),
        SphereByCurvature(10, 0.05),
        SphereByRadius(10, 5),
        Parabola(10.0, A=-0.0),
        Parabola(10.0, A=0.5),
        Parabola(10.0, A=-0.5),
        Conic(10, C=0.1, K=0.1),
        Conic(10, C=-0.1, K=-0.1),
        Asphere(10, C=0.1, K=0.1, alphas=[0.1, 0.01, 0.002]),
        Asphere(10, C=-0.1, K=-0.1, alphas=[0.1, 0.01, 0.002]),
        XYPolynomial(
            10, C=0.1, K=-0.1, coefficients=[[0.1, 0.2, 0.0], [0.01, 0.0, 0.01]]
        ),
    ]

    for module in surfaces_3d:
        check_surface_module_3d(module, False, dtype, device)
        check_surface_module_3d(module.reverse(), False, dtype, device)
