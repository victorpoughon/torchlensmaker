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

import torch

from torchlensmaker.core.functional_kernel import FunctionalKernel
from torchlensmaker.core.tensor_manip import init_param
from torchlensmaker.implicit import ImplicitResult, implicit_disk_2d, implicit_disk_3d
from torchlensmaker.implicit.implicit_plane import implicit_yaxis_2d
from torchlensmaker.kinematics.homogeneous_geometry import (
    hom_identity_2d,
    hom_identity_3d,
)
from torchlensmaker.surfaces.implicit_solver import implicit_surface_local_raytrace
from torchlensmaker.surfaces.sag_surface import SolverConfig, implicit_solver_config
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


def implicit_domain(
    F: BatchTensor,
    points: Batch2DTensor,
    tol: float,
) -> MaskTensor:
    return torch.abs(F) <= tol


def temp_implicit(dim: int, R: torch.Tensor):
    impf = implicit_disk_2d if dim == 2 else implicit_disk_3d

    def f(points: torch.Tensor, *, order: int) -> ImplicitResult:
        res = impf(points, R=R, order=order)
        assert res.grad is not None
        return ImplicitResult(res.val, res.grad, res.hess)

    return f


class ImplicitDiskSurfaceKernel(FunctionalKernel):
    """
    Functional kernel for implicit disk perpendicular to the optical axis.
    In 2D, it reduces to a line segment, sometimes called a plane.
    """

    inputs = {"P": BatchNDTensor, "V": BatchNDTensor, "tf_in": Tf}
    params = {"diameter": ScalarTensor}

    outputs = {
        "t": BatchTensor,
        "normals": BatchNDTensor,
        "valid": MaskTensor,
        "points_local": BatchNDTensor,
        "points_global": BatchNDTensor,
        "rsm": BatchTensor,
    }

    def __init__(self, dim: int, solver_config: SolverConfig):
        self.dim = dim
        self.solver_config = solver_config

    def apply(
        self,
        P: BatchNDTensor,
        V: BatchNDTensor,
        tf_in: Tf,
        diameter: ScalarTensor,
    ) -> tuple[
        BatchTensor,
        BatchNDTensor,
        MaskTensor,
        BatchNDTensor,
        BatchNDTensor,
        BatchTensor,
    ]:
        F = temp_implicit(self.dim, R=diameter / 2)
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
    ) -> tuple[ScalarTensor]:
        return (torch.tensor(10.0, dtype=dtype, device=device),)


class ImplicitDisk(SurfaceElement):
    """
    Disk surface (2D or 3D) implemented as an implicit function
    """

    default_config = SolverConfig(
        implicit_solver="newton",
        num_iter=2,
        damping=0.95,
        tol=1e-4,
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
        self.func2d = ImplicitDiskSurfaceKernel(2, self.solver_config)
        self.func3d = ImplicitDiskSurfaceKernel(3, self.solver_config)

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
        return SurfaceElementOutput(
            *func.apply(P, V, tf, self.diameter), tf.clone(), tf.clone()
        )

    def outer_extent(self, anchor: ScalarTensor) -> ScalarTensor:
        return torch.zeros_like(anchor)

    def render(self) -> Any:
        return {
            "type": "surface-plane",
            "radius": self.diameter.item() / 2,
        }
