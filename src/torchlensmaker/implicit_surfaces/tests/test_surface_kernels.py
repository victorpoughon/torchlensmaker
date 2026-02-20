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
import onnxruntime

from torchlensmaker.implicit_surfaces.surface_sphere_by_curvature import SphereByCurvature2DSurfaceKernel
from torchlensmaker.implicit_surfaces.surface_disk import Disk2DSurfaceKernel
from torchlensmaker.implicit_surfaces.surface_sphere_by_radius import SphereByRadius2DSurfaceKernel

from torchlensmaker.core.functional_kernel import export_onnx, FunctionalKernel

from torchlensmaker.testing.test_functional_kernels_testing import (
    check_kernels_example_inputs_and_params,
    check_kernels_eval,
    check_kernels_export_onnx,
)


kernels_library: Dict[str, FunctionalKernel] = {
    "SphereC2D-1": SphereByCurvature2DSurfaceKernel(1),
    "SphereC2D-6": SphereByCurvature2DSurfaceKernel(6),
    "SphereC2D-12": SphereByCurvature2DSurfaceKernel(12),
    # "SphereR2D": SphereByRadius2DSurfaceKernel(),
    "Disk2D": Disk2DSurfaceKernel(),
}


def test_surface_kernels_inputs_and_params_float32() -> None:
    dtype = torch.float32
    device = torch.device("cpu")

    # Export, load, compare eval on example inputs
    for name, kernel in kernels_library.items():
        check_kernels_example_inputs_and_params(name, kernel, dtype, device)


def test_surface_kernels_inputs_and_params_float64() -> None:
    dtype = torch.float64
    device = torch.device("cpu")

    # Export, load, compare eval on example inputs
    for name, kernel in kernels_library.items():
        check_kernels_example_inputs_and_params(name, kernel, dtype, device)


def test_surface_kernels_eval_float32() -> None:
    dtype = torch.float32
    device = torch.device("cpu")

    # Export, load, compare eval on example inputs
    for name, kernel in kernels_library.items():
        check_kernels_eval(name, kernel, dtype, device)


def test_surface_kernels_eval_float64() -> None:
    dtype = torch.float64
    device = torch.device("cpu")

    # Export, load, compare eval on example inputs
    for name, kernel in kernels_library.items():
        check_kernels_eval(name, kernel, dtype, device)


def test_surface_kernels_export_onnx_float32(tmp_path: Path) -> None:
    dtype = torch.float32
    device = torch.device("cpu")

    # Export, load, compare eval on example inputs
    for name, kernel in kernels_library.items():
        check_kernels_export_onnx(name, kernel, tmp_path, dtype, device)
