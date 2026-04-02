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

from pathlib import Path
from typing import Dict

import onnxruntime
import pytest
import torch

from torchlensmaker.core.functional_kernel import FunctionalKernel
from torchlensmaker.surfaces.surface_asphere import (
    AsphereOuterExtentSurfaceKernel,
    AsphereSurfaceKernel,
)
from torchlensmaker.surfaces.surface_conic import (
    ConicOuterExtentSurfaceKernel,
    ConicSurfaceKernel,
)
from torchlensmaker.surfaces.surface_disk import (
    DiskSurfaceKernel,
)
from torchlensmaker.surfaces.surface_parabola import (
    ParabolaOuterExtentSurfaceKernel,
    ParabolaSurfaceKernel,
)
from torchlensmaker.surfaces.surface_plane import PlaneSurfaceKernel
from torchlensmaker.surfaces.surface_sphere_by_curvature import (
    SphereByCurvatureOuterExtentSurfaceKernel,
    SphereByCurvatureSurfaceKernel,
)
from torchlensmaker.surfaces.surface_sphere_by_radius import (
    SphereByRadiusSurfaceKernel,  # TODO
)
from torchlensmaker.surfaces.surface_xypolynomial import XYPolynomialSurfaceKernel
from torchlensmaker.testing.test_functional_kernels_testing import (
    check_kernels_eval,
    check_kernels_example_inputs_and_params,
    check_kernels_export_onnx,
)

config1 = dict(num_iter=1, damping=1.0, tol=1e-3, lift_function="raw", implicit_solver="newton")
config6 = dict(num_iter=6, damping=0.9, tol=1e-3, lift_function="raw", implicit_solver="newton")
config12 = dict(num_iter=12, damping=0.9, tol=1e-4, lift_function="raw", implicit_solver="newton")

kernels_library: Dict[str, FunctionalKernel] = {
    "SphereC2D-1": SphereByCurvatureSurfaceKernel(2, config1),
    "SphereC2D-6": SphereByCurvatureSurfaceKernel(2, config6),
    "SphereC2D-12": SphereByCurvatureSurfaceKernel(2, config12),
    "SphereC3D-1": SphereByCurvatureSurfaceKernel(3, config1),
    "SphereC3D-6": SphereByCurvatureSurfaceKernel(3, config6),
    "SphereC3D-12": SphereByCurvatureSurfaceKernel(3, config12),
    "SphereCOuterExtent": SphereByCurvatureOuterExtentSurfaceKernel(),
    "Parabola2D-1": ParabolaSurfaceKernel(2, config1),
    "Parabola2D-6": ParabolaSurfaceKernel(2, config1),
    "Parabola3D-1": ParabolaSurfaceKernel(3, config1),
    "Parabola3D-6": ParabolaSurfaceKernel(3, config6),
    "ParabolaOuterExtent": ParabolaOuterExtentSurfaceKernel(),
    "Conic2D-1": ConicSurfaceKernel(2, config1),
    "Conic2D-6": ConicSurfaceKernel(2, config6),
    "Conic3D-1": ConicSurfaceKernel(3, config1),
    "Conic3D-6": ConicSurfaceKernel(3, config6),
    "ConicOuterExtent": ConicOuterExtentSurfaceKernel(),
    "Asphere2D-1": AsphereSurfaceKernel(2, config1),
    "Asphere2D-6": AsphereSurfaceKernel(2, config6),
    "Asphere3D-1": AsphereSurfaceKernel(3, config1),
    "Asphere3D-6": AsphereSurfaceKernel(3, config6),
    "AsphereOuterExtent": AsphereOuterExtentSurfaceKernel(),
    "XYPolynomial3D-1": XYPolynomialSurfaceKernel(config6),
    "XYPolynomial3D-2": XYPolynomialSurfaceKernel(config6),
    # "SphereR2D": SphereByRadius2DSurfaceKernel(),
    "Disk2D": DiskSurfaceKernel(2),
    "Disk3D": DiskSurfaceKernel(3),
    "Plane2D": PlaneSurfaceKernel(2),
    "Plane3D": PlaneSurfaceKernel(3),
}


def test_surface_kernels_inputs_and_params(
    dtype: torch.dtype, device: torch.device
) -> None:
    # Export, load, compare eval on example inputs
    for name, kernel in kernels_library.items():
        check_kernels_example_inputs_and_params(name, kernel, dtype, device)


def test_surface_kernels_eval(dtype: torch.dtype, device: torch.device) -> None:
    # Export, load, compare eval on example inputs
    for name, kernel in kernels_library.items():
        check_kernels_eval(name, kernel, dtype, device)


def test_surface_kernels_export_onnx(
    dtype: torch.dtype, device: torch.device, tmp_path: Path
) -> None:
    # Note this test only works in float32 as of Jan 2026
    # because onnxruntime cpu doesn't seem to support cos() in float64...
    if dtype == torch.float64:
        return

    # Export, load, compare eval on example inputs
    for name, kernel in kernels_library.items():
        check_kernels_export_onnx(name, kernel, tmp_path, dtype, device)
