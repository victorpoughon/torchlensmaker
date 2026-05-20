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
from torchlensmaker.core.tensor_manip import init_param
from torchlensmaker.kinematics.homogeneous_geometry import (
    hom_identity_3d,
)
from torchlensmaker.raytracing.parametric_solver import (
    parametric_surface_local_raytrace,
)
from torchlensmaker.raytracing.parametric_solver_config import (
    InitClosest,
    ParametricSolverConfig,
    make_domain_function,
    make_parametric_solver,
)
from torchlensmaker.raytracing.surface_raytrace import surface_raytrace
from torchlensmaker.types import (
    BatchNDTensor,
    BatchTensor,
    MaskTensor,
    Tf,
)

from .kernels_utils import example_rays_3d
from .surface_element import SurfaceElement, SurfaceRecord


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

    def __init__(
        self,
        degree: tuple[int, int],
        periodic: tuple[bool, bool],
        clamped: tuple[bool, bool],
        solver_config: ParametricSolverConfig,
    ):
        self.degree = degree
        self.solver_config = solver_config
        self.periodic = periodic
        self.clamped = clamped
        self.solver = make_parametric_solver(solver_config)

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
            return tnodo.bspline_surface(
                uv,
                control_points,
                degree=self.degree,
                order=(order, order),
                periodic=self.periodic,
                clamped=self.clamped,
            )

        local_solver = partial(
            parametric_surface_local_raytrace,
            parametric_function=S,
            solver=self.solver,
            domain_function=make_domain_function(self.solver_config, S),
        )

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
        # (K, L, 3) control point grid, flat surface at x=0 (perpendicular to optical axis)
        g = torch.linspace(-1, 1, 4, dtype=dtype, device=device)
        gu, gv = torch.meshgrid(g, g, indexing="ij")
        gx = torch.zeros_like(gu)
        control_points = torch.stack([gx, gu, gv], dim=-1)  # (4, 4, 3)
        return (control_points,)


class BSplineSurface(SurfaceElement):
    """
    BSpline surface with 3D control points
    """

    default_config: ParametricSolverConfig = {
        "parametric_solver": "newton",
        "num_iter": 10,
        "damping": 1.0,
        "tol": 1e-4,
        "init": InitClosest(),
        "clamp_positive": False,
        "singular_check": False,
        "periodic_uv": (False, False),
        "u_epsilon": 0.0,
        "v_epsilon": 0.0,
    }

    def __init__(
        self,
        control_points: torch.Tensor | torch.nn.Parameter,
        *,
        trainable: bool = False,
        periodic: tuple[bool, bool] = (False, False),
        clamped: tuple[bool, bool] = (True, True),
        solver_config: dict[str, Any] = {},
    ):
        super().__init__()

        self.solver_config = ParametricSolverConfig(
            **self.default_config | solver_config
        )

        # TODO degree hardcoded for now
        # it must be a static kernel parameter
        self.degree = (2, 2)
        self.periodic = periodic
        self.clamped = clamped

        self.control_points = init_param(
            self, "control_points", control_points, trainable
        )
        self.func3d = BSplineSurfaceKernel(
            self.degree,
            self.periodic,
            self.clamped,
            self.solver_config,
        )

    def clone(self, **overrides: Any) -> Self:
        kwargs: dict[str, Any] = dict(
            control_points=self.control_points,
            trainable=self.control_points.requires_grad,
            periodic=self.periodic,
            clamped=self.clamped,
            solver_config=self.solver_config,
        )
        return type(self)(**kwargs | overrides)

    def __repr__(self) -> str:
        return f"{self._get_name()}(control_points={self.control_points.tolist()})"

    def reverse(self) -> Self:
        return self.clone()

    def forward(self, P: BatchTensor, V: BatchTensor, tf: Tf) -> SurfaceRecord:
        t, normal, valid, points_local, points_global, rsm = self.func3d.apply(
            P, V, tf, self.control_points
        )

        return SurfaceRecord(
            t, normal, valid, points_local, points_global, rsm, tf.clone(), tf.clone()
        )

    def render(self, matrix: torch.Tensor) -> tlmv.SurfaceBSpline:
        return tlmv.SurfaceBSpline(
            points=self.control_points.tolist(),
            degree=self.degree,
            periodic=self.periodic,
            clamped=self.clamped,
            samples=(100, 100),  # TODO
            matrix=matrix.tolist(),
        )
