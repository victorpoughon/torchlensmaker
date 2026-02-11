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


from typing import Dict
from pathlib import Path
from itertools import chain

import pytest

import torch
import torch.nn as nn
import onnxruntime

from torchlensmaker.kinematics.homogeneous_geometry import hom_identity_2d

from torchlensmaker.implicit_surfaces.surface_elements import SphereC

from torchlensmaker.testing.test_utils import (
    check_model_eval,
    check_model_eval_and_grad,
)


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
    dfk, ifk = hom_identity_2d(dtype, device)

    if trainable:
        t, normals, valid, dfk_out, ifk_out = check_model_eval_and_grad(
            mod, (P, V, dfk, ifk), allow_none_grad
        )
    else:
        t, normals, valid, dfk_out, ifk_out = check_model_eval(mod, (P, V, dfk, ifk))

    # Check output is sane
    assert t.shape == (N,)
    assert normals.shape == (N, 2)
    assert valid.shape == (N,)
    assert dfk_out.shape == ifk_out.shape == (3, 3)

    assert t.device == device
    assert normals.device == device
    assert valid.device == device
    assert dfk_out.device == device
    assert ifk_out.device == device


def test_sag_surfaces_modules() -> None:
    dtype, device = torch.float64, torch.device("cpu")
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)

    surfaces_2d = [
        SphereC(10.0, C=0.0),
        SphereC(10.0, C=0.5),
        SphereC(10.0, C=-0.5),
    ]

    for module in surfaces_2d:
        check_surface_module_2d(module, False, dtype, device)
