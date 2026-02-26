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

import pytest

import torch
import onnxruntime

from torchlensmaker.implicit_surfaces.surface_sphere_by_curvature import (
    SphereByCurvatureSurfaceKernel,
)
from torchlensmaker.implicit_surfaces.surface_disk import (
    DiskSurfaceKernel,
)
from torchlensmaker.implicit_surfaces.surface_parabola import ParabolaSurfaceKernel
from torchlensmaker.implicit_surfaces.surface_sphere_by_radius import (
    SphereByRadiusSurfaceKernel, # TODO
)
from torchlensmaker.implicit_surfaces.surface_conic import ConicSurfaceKernel
from torchlensmaker.implicit_surfaces.surface_asphere import AsphereSurfaceKernel
from torchlensmaker.implicit_surfaces.surface_xypolynomial import XYPolynomialSurfaceKernel

from torchlensmaker.core.functional_kernel import FunctionalKernel

from torchlensmaker.testing.test_functional_kernels_testing import (
    check_kernels_example_inputs_and_params,
    check_kernels_eval,
    check_kernels_export_onnx,
)


kernels_library: Dict[str, FunctionalKernel] = {
    "SphereC2D-1": SphereByCurvatureSurfaceKernel(2, 1, 0.9, 1e-3),
    "SphereC2D-6": SphereByCurvatureSurfaceKernel(2, 6, 1.0, 1e-6),
    "SphereC2D-12": SphereByCurvatureSurfaceKernel(2, 12, 0.5, 1e-12),
    "SphereC3D-1": SphereByCurvatureSurfaceKernel(3, 1, 1.0, 1e-3),
    "SphereC3D-6": SphereByCurvatureSurfaceKernel(3, 6, 0.9, 1e-6),
    "SphereC3D-12": SphereByCurvatureSurfaceKernel(3, 12, 0.9, 1e-6),
    "Parabola2D-1": ParabolaSurfaceKernel(2, 1, 1.0, 1e-3),
    "Parabola2D-6": ParabolaSurfaceKernel(2, 1, 1.0, 1e-3),
    "Parabola3D-1": ParabolaSurfaceKernel(3, 1, 1.0, 1e-3),
    "Parabola3D-6": ParabolaSurfaceKernel(3, 6, 1.0, 1e-3),
    "Conic2D-1": ConicSurfaceKernel(2, 1, 1.0, 1e-3),
    "Conic2D-6": ConicSurfaceKernel(2, 6, 1.0, 1e-3),
    "Conic3D-1": ConicSurfaceKernel(3, 1, 1.0, 1e-3),
    "Conic3D-6": ConicSurfaceKernel(3, 6, 1.0, 1e-3),
    "Asphere2D-1": AsphereSurfaceKernel(2, 1, 1.0, 1e-3),
    "Asphere2D-6": AsphereSurfaceKernel(2, 6, 1.0, 1e-3),
    "Asphere3D-1": AsphereSurfaceKernel(3, 1, 1.0, 1e-3),
    "Asphere3D-6": AsphereSurfaceKernel(3, 6, 1.0, 1e-3),
    "XYPolynomial3D-1": XYPolynomialSurfaceKernel(6, 1.0, 1e-3),
    "XYPolynomial3D-2": XYPolynomialSurfaceKernel(6, 1.0, 1e-3),
    # "SphereR2D": SphereByRadius2DSurfaceKernel(),
    "Disk2D": DiskSurfaceKernel(2),
    "Disk3D": DiskSurfaceKernel(3),
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
