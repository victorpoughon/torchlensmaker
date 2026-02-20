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


from typing import Dict, Any
from pathlib import Path
from itertools import chain

import pytest

import torch
import torch.nn as nn
import onnxruntime

from torchlensmaker.kinematics.homogeneous_geometry import hom_identity_2d, hom_identity_3d
from torchlensmaker.types import BatchTensor, Batch2DTensor, Tf2D
from torchlensmaker.implicit_surfaces.surface_sphere_by_curvature import (
    SphereByCurvature,
)
from torchlensmaker.implicit_surfaces.surface_disk import Disk


def check_model_eval(
    model: nn.Module, inputs: tuple[Batch2DTensor, Batch2DTensor, Tf2D]
) -> Any:
    "Evaluate a model forwards and run sanity checks"

    # Check the forward pass
    t, normals, valid, tf_surface, tf_next = model(*inputs)
    assert t.isfinite().all()
    assert normals.isfinite().all()
    assert valid.isfinite().all()
    assert tf_surface.direct.isfinite().all()
    assert tf_surface.inverse.isfinite().all()
    assert tf_next.direct.isfinite().all()
    assert tf_next.inverse.isfinite().all()

    return t, normals, valid, tf_surface, tf_next


def check_model_eval_and_grad(
    model: nn.Module,
    inputs: tuple[Batch2DTensor, Batch2DTensor, Tf2D],
    allow_none_grad: bool = False,
) -> Any:
    """
    Evaluate a model forwards and backwards and run sanity checks
    Expects at least one trainable parameter
    """

    # Check the forward pass
    t, normals, valid, tf_surface, tf_next = model(*inputs)
    assert t.isfinite().all()
    assert normals.isfinite().all()
    assert valid.isfinite().all()
    assert tf_surface.direct.isfinite().all()
    assert tf_surface.inverse.isfinite().all()
    assert tf_next.direct.isfinite().all()
    assert tf_next.inverse.isfinite().all()

    # Check the backward pass
    parameters = list(model.named_parameters())
    assert len(parameters) > 0

    loss = t.pow(2).sum() + tf_next.direct.sum().pow(2)
    model.zero_grad()
    loss.backward()  # type: ignore[no-untyped-call]
    for name, param in parameters:
        print(f"grad({name}) = {param.grad}")
        assert allow_none_grad or param.grad is not None
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), (
                f"Gradient of {name} contains NaN or Inf: {param.grad}"
            )

    return t, normals, valid, tf_surface, tf_next


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
        t, normals, valid, tf_surface, tf_next = check_model_eval_and_grad(
            mod, (P, V, tfid), allow_none_grad
        )
    else:
        t, normals, valid, tf_surface, tf_next = check_model_eval(mod, (P, V, tfid))

    # Check output is sane
    assert t.shape == (N,)
    assert normals.shape == (N, 2)
    assert valid.shape == (N,)
    assert tf_surface.shape == (3, 3)
    assert tf_next.shape == (3, 3)

    assert t.device == device
    assert normals.device == device
    assert valid.device == device
    assert tf_surface.device == device
    assert tf_next.device == device


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
        t, normals, valid, tf_surface, tf_next = check_model_eval_and_grad(
            mod, (P, V, tfid), allow_none_grad
        )
    else:
        t, normals, valid, tf_surface, tf_next = check_model_eval(mod, (P, V, tfid))

    # Check output is sane
    assert t.shape == (N,)
    assert normals.shape == (N, 3)
    assert valid.shape == (N,)
    assert tf_surface.shape == (4, 4)
    assert tf_next.shape == (4, 4)

    assert t.device == device
    assert normals.device == device
    assert valid.device == device
    assert tf_surface.device == device
    assert tf_next.device == device


def test_sag_surfaces_modules_2d() -> None:
    dtype, device = torch.float64, torch.device("cpu")
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)

    surfaces_2d = [
        SphereByCurvature(10.0, C=0.0),
        SphereByCurvature(10.0, C=0.5),
        SphereByCurvature(10.0, C=-0.5),
        Disk(10.0),
    ]

    for module in surfaces_2d:
        check_surface_module_2d(module, False, dtype, device)


def test_sag_surfaces_modules_3d() -> None:
    dtype, device = torch.float64, torch.device("cpu")
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)

    surfaces_3d = [
        Disk(10.0),
        SphereByCurvature(10, 0.05),
    ]

    for module in surfaces_3d:
        check_surface_module_3d(module, False, dtype, device)
