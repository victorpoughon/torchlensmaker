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

from torchlensmaker.surfaces.surface_asphere import (
    AsphereOuterExtentSurfaceKernel,
    AsphereSurfaceKernel,
)
from torchlensmaker.surfaces.surface_conic import (
    ConicOuterExtentSurfaceKernel,
    ConicSurfaceKernel,
)
from torchlensmaker.surfaces.surface_disk import DiskSurfaceKernel
from torchlensmaker.surfaces.surface_parabola import (
    ParabolaOuterExtentSurfaceKernel,
    ParabolaSurfaceKernel,
)
from torchlensmaker.surfaces.surface_plane import PlaneSurfaceKernel
from torchlensmaker.surfaces.surface_point import PointSurfaceKernel
from torchlensmaker.surfaces.surface_sphere import SphereSurfaceKernel
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
    pytest.param(SphereByCurvatureSurfaceKernel(2, config1), id="SphereC2D-1"),
    pytest.param(SphereByCurvatureSurfaceKernel(2, config6), id="SphereC2D-6"),
    pytest.param(SphereByCurvatureSurfaceKernel(2, config12), id="SphereC2D-12"),
    pytest.param(SphereByCurvatureSurfaceKernel(3, config1), id="SphereC3D-1"),
    pytest.param(SphereByCurvatureSurfaceKernel(3, config6), id="SphereC3D-6"),
    pytest.param(SphereByCurvatureSurfaceKernel(3, config12), id="SphereC3D-12"),
    pytest.param(SphereByCurvatureOuterExtentSurfaceKernel(), id="SphereCOuterExtent"),
    pytest.param(ParabolaSurfaceKernel(2, config1), id="Parabola2D-1"),
    pytest.param(ParabolaSurfaceKernel(2, config6), id="Parabola2D-6"),
    pytest.param(ParabolaSurfaceKernel(3, config1), id="Parabola3D-1"),
    pytest.param(ParabolaSurfaceKernel(3, config6), id="Parabola3D-6"),
    pytest.param(ParabolaOuterExtentSurfaceKernel(), id="ParabolaOuterExtent"),
    pytest.param(ConicSurfaceKernel(2, config1), id="Conic2D-1"),
    pytest.param(ConicSurfaceKernel(2, config6), id="Conic2D-6"),
    pytest.param(ConicSurfaceKernel(3, config1), id="Conic3D-1"),
    pytest.param(ConicSurfaceKernel(3, config6), id="Conic3D-6"),
    pytest.param(ConicOuterExtentSurfaceKernel(), id="ConicOuterExtent"),
    pytest.param(AsphereSurfaceKernel(2, config1), id="Asphere2D-1"),
    pytest.param(AsphereSurfaceKernel(2, config6), id="Asphere2D-6"),
    pytest.param(AsphereSurfaceKernel(3, config1), id="Asphere3D-1"),
    pytest.param(AsphereSurfaceKernel(3, config6), id="Asphere3D-6"),
    pytest.param(AsphereOuterExtentSurfaceKernel(), id="AsphereOuterExtent"),
    pytest.param(XYPolynomialSurfaceKernel(config6), id="XYPolynomial3D-1"),
    pytest.param(XYPolynomialSurfaceKernel(config6), id="XYPolynomial3D-2"),
    # pytest.param(SphereByRadiusSurfaceKernel(...),          id="SphereR2D"),
    pytest.param(SphereSurfaceKernel(2, config_sphere), id="Sphere2D"),
    pytest.param(SphereSurfaceKernel(3, config_sphere), id="Sphere3D"),
    pytest.param(DiskSurfaceKernel(2), id="Disk2D"),
    pytest.param(DiskSurfaceKernel(3), id="Disk3D"),
    pytest.param(PlaneSurfaceKernel(2), id="Plane2D"),
    pytest.param(PlaneSurfaceKernel(3), id="Plane3D"),
    pytest.param(PointSurfaceKernel(2), id="Point2D"),
    pytest.param(PointSurfaceKernel(3), id="Point3D"),
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
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    # Note this test only works in float32 as of Jan 2026
    # because onnxruntime cpu doesn't seem to support cos() in float64...
    if dtype == torch.float64:
        return
    check_kernels_export_onnx(request.node.callspec.id, kernel, tmp_path, dtype, device)
