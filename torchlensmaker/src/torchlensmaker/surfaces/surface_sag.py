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
import torch.nn as nn
import torchimplicit as ti
from jaxtyping import Bool, Float

from torchlensmaker.core.functional_kernel import FunctionalKernel
from torchlensmaker.core.tensor_manip import init_param
from torchlensmaker.kinematics.homogeneous_geometry import (
    hom_identity_2d,
    hom_identity_3d,
)
from torchlensmaker.surfaces.implicit_solver import implicit_solver_newton
from torchlensmaker.surfaces.sag_geometry import (
    lens_diameter_implicit_domain_2d,
    lens_diameter_implicit_domain_3d,
)
from torchlensmaker.surfaces.sag_surface import (
    SolverConfig,
    sag_solver_config,
    sag_surface_raytrace,
)
from torchlensmaker.surfaces.surface_anchor import SurfaceScaleAnchorKernel
from torchlensmaker.types import (
    BatchNDTensor,
    BatchTensor,
    MaskTensor,
    ScalarTensor,
    Tf,
)

from .kernels_utils import example_rays_2d, example_rays_3d
from .surface_element import SurfaceElement, SurfaceElementOutput


class SagSurfaceKernel(FunctionalKernel):
    """
    Generic 2D/3D sag surface kernel
    """

    inputs = {"P": BatchNDTensor, "V": BatchNDTensor, "tf_in": Tf}

    params = {
        "diameter": ScalarTensor,
        "normalize": Bool[torch.Tensor, ""],
        "params": torch.Tensor,
    }

    outputs = {
        "t": BatchTensor,
        "normals": BatchNDTensor,
        "valid": MaskTensor,
        "points_local": BatchNDTensor,
        "points_global": BatchNDTensor,
        "rsm": BatchTensor,
    }

    def __init__(self, dim: int, sag: ti.SagFunction, solver_config: SolverConfig):
        self.dim = dim
        self.sag = sag
        self.solver_config = solver_config

    def apply(
        self,
        P: BatchNDTensor,
        V: BatchNDTensor,
        tf_in: Tf,
        diameter: ScalarTensor,
        normalize: Bool[torch.Tensor, ""],
        params: torch.Tensor,
    ) -> tuple[
        BatchTensor,
        BatchNDTensor,
        MaskTensor,
        BatchNDTensor,
        BatchNDTensor,
        BatchTensor,
    ]:
        liftf, domainf, implicit_solver = sag_solver_config(
            self.dim, self.solver_config, diameter
        )

        # Bind sag function parameters for the solver
        def S(points: BatchNDTensor, *, order: int) -> ti.SagResult:
            return self.sag(points, params=params, order=order)

        return sag_surface_raytrace(
            S,
            liftf,
            domainf,
            implicit_solver,
            P,
            V,
            tf_in,
            diameter,
            normalize,
        )

    def example_inputs(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[BatchNDTensor, BatchNDTensor, Tf]:
        tf: Tf
        if self.dim == 2:
            P, V = example_rays_2d(10, dtype, device)
            tf = hom_identity_2d(dtype, device)
        else:
            P, V = example_rays_3d(10, dtype, device)
            tf = hom_identity_3d(dtype, device)

        return P, V, tf

    # TODO store example params per sag function
    def example_params(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[
        ScalarTensor,
        ScalarTensor,
        Bool[torch.Tensor, ""],
    ]:
        return (
            torch.tensor(10.0, dtype=dtype, device=device),
            torch.tensor(0.5, dtype=dtype, device=device),
            torch.tensor(False, dtype=torch.bool, device=device),
        )
