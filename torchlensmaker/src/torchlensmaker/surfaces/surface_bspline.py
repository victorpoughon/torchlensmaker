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

from functools import partial
from typing import Any, Self

import tlmviewer as tlmv
import torch
import torchnodo as tnodo

from torchlensmaker.core.functional_kernel import FunctionalKernel
from torchlensmaker.core.geometry import unit_vector
from torchlensmaker.core.tensor_manip import init_param
from torchlensmaker.kinematics.homogeneous_geometry import (
    hom_identity_2d,
    hom_identity_3d,
)
from torchlensmaker.surfaces.parametric_solver import parametric_surface_local_raytrace
from torchlensmaker.types import (
    Batch2DTensor,
    Batch3DTensor,
    BatchNDTensor,
    BatchTensor,
    MaskTensor,
    ScalarTensor,
    Tf,
)

from .kernels_utils import example_rays_2d, example_rays_3d
from .raytrace import surface_raytrace
from .sag_geometry import lens_diameter_domain_2d, lens_diameter_domain_3d
from .surface_element import SurfaceElement, SurfaceElementOutput


class BSplineSurfaceKernel(FunctionalKernel):
    """
    Functional kernel for a BSpline surface with 3D control points
    """

    inputs = {
        "P": BatchNDTensor,
        "V": BatchNDTensor,
        "tf_in": Tf,
    }

    params = {
        "control_points": torch.Tensor,
    }

    outputs = {
        "t": BatchTensor,
        "normals": BatchNDTensor,
        "valid": MaskTensor,
        "points_local": BatchNDTensor,
        "points_global": BatchNDTensor,
        "rsm": BatchTensor,
    }

    def __init__(self, degree: tuple[int, int]):
        self.degree = degree

    def apply(
        self,
        P: BatchNDTensor,
        V: BatchNDTensor,
        tf_in: Tf,
        control_points: torch.Tensor,
    ) -> tuple[
        BatchTensor,
        BatchNDTensor,
        MaskTensor,
        BatchNDTensor,
        BatchNDTensor,
        BatchTensor,
    ]:
        def S(uv: torch.Tensor, *, order: int) -> torch.Tensor:
            # TODO order
            res = tnodo.bspline_surface(
                uv,
                control_points,
                degree=self.degree,
                order=(2, 2),
                periodic=(False, False),
                clamped=(True, True),
            )
            return res

        local_solver = partial(parametric_surface_local_raytrace, parametric_function=S)

        return surface_raytrace(P, V, tf_in, local_solver)

    def example_inputs(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[BatchNDTensor, BatchNDTensor, Tf]:
        P, V = example_rays_3d(10, dtype, device)
        tf = hom_identity_3d(dtype, device)
        return P, V, tf

    def example_params(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor]:
        # (N, 3) control points
        g = torch.linspace(0, 1, 10, dtype=dtype, device=device)
        control_points = torch.stack((g, g, g))
        return (control_points,)


class BSplineSurface(SurfaceElement):
    """
    BSpline surface with 3D control points
    """

    def __init__(
        self,
        control_points: torch.Tensor | torch.nn.Parameter,
        *,
        trainable: bool = False,
        solver_config: dict[str, Any] = {},
    ):
        super().__init__()
        self.solver_config = {}  # TODO

        # TODO degree hardcoded for now
        # it must be a static kernel parameter
        self.degree = (2, 2)

        self.control_points = init_param(
            self, "control_points", control_points, trainable
        )
        self.func3d = BSplineSurfaceKernel(self.degree)

    def clone(self, **overrides: Any) -> Self:
        kwargs: dict[str, Any] = dict(
            control_points=self.control_points,
            trainable=self.control_points.requires_grad,
            solver_config=self.solver_config,
        )
        return type(self)(**kwargs | overrides)

    def __repr__(self) -> str:
        return f"{self._get_name()}(control_points={self.control_points.tolist()})"

    def reverse(self) -> Self:
        return self.clone()

    def forward(self, P: BatchTensor, V: BatchTensor, tf: Tf) -> SurfaceElementOutput:
        t, normal, valid, points_local, points_global, rsm = self.func3d.apply(
            P, V, tf, self.control_points
        )

        return SurfaceElementOutput(
            t, normal, valid, points_local, points_global, rsm, tf.clone(), tf.clone()
        )

    def render(self, matrix: torch.Tensor) -> tlmv.SurfaceBSpline:
        return tlmv.SurfaceBSpline(
            points=self.control_points.tolist(),
            degree=self.degree,
            knot_type="clamped",
            samples=(30, 30),  # TODO
            matrix=matrix.tolist(),
        )
