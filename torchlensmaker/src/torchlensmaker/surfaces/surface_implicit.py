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
import torchimplicit as ti

from torchlensmaker.core.functional_kernel import FunctionalKernel
from torchlensmaker.core.tensor_manip import init_param
from torchlensmaker.kinematics.homogeneous_geometry import (
    hom_identity_2d,
    hom_identity_3d,
)
from torchlensmaker.surfaces.implicit_solver import implicit_surface_local_raytrace
from torchlensmaker.surfaces.sag_surface import SolverConfig, implicit_solver_config
from torchlensmaker.types import (
    BatchNDTensor,
    BatchTensor,
    MaskTensor,
    ScalarTensor,
    Tf,
)

from .kernels_utils import example_rays_2d, example_rays_3d
from .raytrace import surface_raytrace
from .sag_geometry import (
    implicit_domain,
)
from .surface_element import SurfaceElement, SurfaceElementOutput


class ImplicitSurfaceKernel(FunctionalKernel):
    """
    Generic 2D/3D implicit surface kernel
    """

    inputs = {"P": BatchNDTensor, "V": BatchNDTensor, "tf_in": Tf}
    params = {"params": torch.Tensor}

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
        dim: int,
        func: ti.ImplicitFunction,
        solver_config: SolverConfig,
    ):
        self.dim = dim
        self.func = func
        self.solver_config = solver_config

    def apply(
        self,
        P: BatchNDTensor,
        V: BatchNDTensor,
        tf_in: Tf,
        params: torch.Tensor,
    ) -> tuple[
        BatchTensor,
        BatchNDTensor,
        MaskTensor,
        BatchNDTensor,
        BatchNDTensor,
        BatchTensor,
    ]:
        # Bind function parameters for the solver
        def F(points: torch.Tensor, *, order: int) -> ti.ImplicitResult:
            res = self.func(points, params, order=order)
            return ti.ImplicitResult(res.val, res.grad, res.hess)

        implicit_solver = implicit_solver_config(self.dim, self.solver_config)
        local_solver = partial(
            implicit_surface_local_raytrace,
            implicit_function=F,
            domain_function=partial(implicit_domain, tol=self.solver_config["tol"]),
            implicit_solver=implicit_solver,
        )

        return surface_raytrace(P, V, tf_in, local_solver)

    def example_inputs(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[BatchNDTensor, BatchNDTensor, Tf]:
        if self.dim == 2:
            P, V = example_rays_2d(10, dtype, device)
            tf = hom_identity_2d(dtype, device)
        else:
            P, V = example_rays_3d(10, dtype, device)
            tf = hom_identity_3d(dtype, device)
        return P, V, tf

    def example_params(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor]:
        return (self.func.example_params(dtype, device),)


class ImplicitDisk(SurfaceElement):
    """
    Disk surface (2D or 3D) implemented as an implicit function
    """

    default_config = SolverConfig(
        implicit_solver="newton",
        num_iter=2,
        damping=0.95,
        tol=1e-4,
        init="closest",
        clamp_positive=True,
    )

    def __init__(
        self,
        diameter: float | ScalarTensor,
        trainable: bool = False,
        solver_config: dict[str, Any] = {},
    ):
        super().__init__()
        self.solver_config = SolverConfig(**self.default_config | solver_config)
        self.diameter = init_param(self, "diameter", diameter, trainable)
        self.func2d = ImplicitSurfaceKernel(2, ti.disk_2d, self.solver_config)
        self.func3d = ImplicitSurfaceKernel(3, ti.disk_3d, self.solver_config)

    def clone(self, **overrides: Any) -> Self:
        kwargs: dict[str, Any] = dict(
            diameter=self.diameter, solver_config=self.solver_config
        )
        return type(self)(**kwargs | overrides)

    def __repr__(self) -> str:
        return f"{self._get_name()}(diameter={self.diameter.item()})"

    def reverse(self) -> Self:
        return self.clone()

    def forward(
        self, P: BatchNDTensor, V: BatchNDTensor, tf: Tf
    ) -> SurfaceElementOutput:
        func = self.func2d if P.shape[-1] == 2 else self.func3d
        params = (self.diameter / 2).unsqueeze(0)
        return SurfaceElementOutput(
            *func.apply(P, V, tf, params), tf.clone(), tf.clone()
        )

    def outer_extent(self, anchor: ScalarTensor) -> ScalarTensor:
        return torch.zeros_like(anchor)

    def render(self, matrix: torch.Tensor) -> tlmv.SurfaceDisk:
        return tlmv.SurfaceDisk(radius=self.diameter.item() / 2, matrix=matrix.tolist())


class Sphere(SurfaceElement):
    """
    Sphere surface (2D or 3D) centered at origin, parameterized by radius R.
    """

    default_config = SolverConfig(
        implicit_solver="newton",
        num_iter=6,
        damping=0.95,
        tol=1e-4,
        init="0",
        clamp_positive=True,
    )

    def __init__(
        self,
        R: float | ScalarTensor | torch.nn.Parameter,
        *,
        trainable: bool = False,
        solver_config: dict[str, Any] = {},
    ):
        super().__init__()
        self.solver_config = SolverConfig(**self.default_config | solver_config)
        self.R = init_param(self, "R", R, trainable)
        self.func2d = ImplicitSurfaceKernel(2, ti.sphere_2d, self.solver_config)
        self.func3d = ImplicitSurfaceKernel(3, ti.sphere_3d, self.solver_config)

    def clone(self, **overrides: Any) -> Self:
        kwargs: dict[str, Any] = dict(
            R=self.R,
            trainable=self.R.requires_grad,
            solver_config=self.solver_config,
        )
        return type(self)(**kwargs | overrides)

    def __repr__(self) -> str:
        return f"{self._get_name()}(R={self.R.item()})"

    def reverse(self) -> Self:
        return self.clone()

    def outer_extent(self, anchor: ScalarTensor) -> ScalarTensor:
        return torch.zeros_like(anchor)

    def forward(self, P: BatchTensor, V: BatchTensor, tf: Tf) -> SurfaceElementOutput:
        func = self.func2d if P.shape[-1] == 2 else self.func3d
        params = self.R.unsqueeze(0)
        return SurfaceElementOutput(
            *func.apply(P, V, tf, params), tf.clone(), tf.clone()
        )

    def render(self, matrix: torch.Tensor) -> tlmv.SurfaceSphere:
        return tlmv.SurfaceSphere(
            R=self.R.item(),
            matrix=matrix.tolist(),
        )
