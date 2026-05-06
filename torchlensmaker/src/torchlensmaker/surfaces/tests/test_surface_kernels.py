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

import onnxruntime
import pytest
import torch
import torchimplicit as ti

from torchlensmaker.surfaces.surface_disk import DiskSurfaceKernel
from torchlensmaker.surfaces.surface_implicit import ImplicitSurfaceKernel
from torchlensmaker.surfaces.surface_plane import PlaneSurfaceKernel
from torchlensmaker.surfaces.surface_point import PointSurfaceKernel
from torchlensmaker.surfaces.surface_sag import (
    SagOuterExtentSurfaceKernel,
    SagSurfaceKernel,
)
from torchlensmaker.surfaces.surface_sphere_by_radius import (
    SphereByRadiusSurfaceKernel,  # TODO
)
from torchlensmaker.testing.test_functional_kernels_testing import (
    check_kernels_eval,
    check_kernels_example_inputs_and_params,
    check_kernels_export_onnx,
)

config1 = dict(
    num_iter=1,
    damping=1.0,
    tol=1e-3,
    lift_function="raw",
    implicit_solver="newton",
    init="closest",
    clamp_positive=True,
)
config6 = dict(
    num_iter=6,
    damping=0.9,
    tol=1e-3,
    lift_function="raw",
    implicit_solver="newton",
    init="closest",
    clamp_positive=True,
)
config12 = dict(
    num_iter=12,
    damping=0.9,
    tol=1e-4,
    lift_function="raw",
    implicit_solver="newton",
    init="closest",
    clamp_positive=True,
)
config_sphere = dict(
    num_iter=6,
    damping=0.95,
    tol=1e-4,
    implicit_solver="newton",
    init="0",
    clamp_positive=True,
)

kernels_cases = [
    # pytest.param(SphereByRadiusSurfaceKernel(...),          id="SphereR2D"),
    pytest.param(DiskSurfaceKernel(2), id="Disk2D"),
    pytest.param(DiskSurfaceKernel(3), id="Disk3D"),
    pytest.param(PlaneSurfaceKernel(2), id="Plane2D"),
    pytest.param(PlaneSurfaceKernel(3), id="Plane3D"),
    pytest.param(PointSurfaceKernel(2), id="Point2D"),
    pytest.param(PointSurfaceKernel(3), id="Point3D"),
    pytest.param(ImplicitSurfaceKernel(2, ti.disk_2d, config1), id="ImplicitDisk2D-1"),
    pytest.param(ImplicitSurfaceKernel(3, ti.disk_3d, config1), id="ImplicitDisk3D-1"),
    pytest.param(
        ImplicitSurfaceKernel(2, ti.sphere_2d, config_sphere), id="ImplicitSphere2D-1"
    ),
    pytest.param(
        ImplicitSurfaceKernel(3, ti.sphere_3d, config_sphere), id="ImplicitSphere3D-1"
    ),
    pytest.param(
        SagSurfaceKernel(2, ti.spherical_sag_2d, config6), id="SagSpherical2D"
    ),
    pytest.param(
        SagSurfaceKernel(3, ti.spherical_sag_3d, config6), id="SagSpherical3D"
    ),
    pytest.param(
        SagSurfaceKernel(2, ti.parabolic_sag_2d, config6), id="SagParabolic2D"
    ),
    pytest.param(
        SagSurfaceKernel(3, ti.parabolic_sag_3d, config6), id="SagParabolic3D"
    ),
    pytest.param(SagSurfaceKernel(2, ti.conical_sag_2d, config6), id="SagConical2D"),
    pytest.param(SagSurfaceKernel(3, ti.conical_sag_3d, config6), id="SagConical3D"),
    pytest.param(SagSurfaceKernel(2, ti.aspheric_sag_2d, config6), id="SagAspheric2D"),
    pytest.param(SagSurfaceKernel(3, ti.aspheric_sag_3d, config6), id="SagAspheric3D"),
    pytest.param(
        SagSurfaceKernel(3, ti.xypolynomial_sag_3d, config6), id="SagXYPolynomial3D"
    ),
    pytest.param(
        SagOuterExtentSurfaceKernel(ti.spherical_sag_2d), id="SagOuterExtentSpherical2D"
    ),
    pytest.param(
        SagOuterExtentSurfaceKernel(ti.parabolic_sag_2d), id="SagOuterExtentParabolic2D"
    ),
    pytest.param(
        SagOuterExtentSurfaceKernel(ti.conical_sag_2d), id="SagOuterExtentConical2D"
    ),
    pytest.param(
        SagOuterExtentSurfaceKernel(ti.aspheric_sag_2d), id="SagOuterExtentAspheric2D"
    ),
]


@pytest.mark.parametrize("kernel", kernels_cases)
def test_surface_kernels_inputs_and_params(
    kernel, dtype: torch.dtype, device: torch.device
) -> None:
    check_kernels_example_inputs_and_params("kernel", kernel, dtype, device)


@pytest.mark.parametrize("kernel", kernels_cases)
def test_surface_kernels_eval(kernel, dtype: torch.dtype, device: torch.device) -> None:
    check_kernels_eval("kernel", kernel, dtype, device)


@pytest.mark.parametrize("kernel", kernels_cases)
def test_surface_kernels_export_onnx(
    kernel,
    dtype: torch.dtype,
    device: torch.device,
    onnx_output_dir: Path,
    request: pytest.FixtureRequest,
) -> None:
    # Note this test only works in float32 as of Jan 2026
    # because onnxruntime cpu doesn't seem to support cos() in float64...
    if dtype == torch.float64:
        return
    check_kernels_export_onnx(
        request.node.callspec.id, kernel, onnx_output_dir, dtype, device
    )
