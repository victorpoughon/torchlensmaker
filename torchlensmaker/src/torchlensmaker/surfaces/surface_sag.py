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

import torch
import torchimplicit as ti
from jaxtyping import Bool

from torchlensmaker.core.functional_kernel import FunctionalKernel
from torchlensmaker.kinematics.homogeneous_geometry import (
    hom_identity_2d,
    hom_identity_3d,
)
from torchlensmaker.types import (
    BatchNDTensor,
    BatchTensor,
    MaskTensor,
    ScalarTensor,
    Tf,
)

from .implicit_solver import (
    ImplicitSolver,
    implicit_surface_local_raytrace,
)
from .kernels_utils import example_rays_2d, example_rays_3d
from .raytrace import surface_raytrace
from .implicit_solver_config import (
    ImplicitSolverConfig,
    make_domain_function_2d,
    make_domain_function_3d,
    make_implicit_solver,
    make_lift_function_2d,
    make_lift_function_3d,
)


def sag_solver_config(
    dim: int, config: ImplicitSolverConfig, diameter: ScalarTensor
) -> tuple[ti.LiftFunction, ti.DomainFunction, ImplicitSolver]:
    """
    Configure a sag function from static parameters
    """

    if dim == 2:
        liftf = make_lift_function_2d(config)
        domainf = make_domain_function_2d(config, diameter)
        implicit_solver = make_implicit_solver(config)
        return liftf, domainf, implicit_solver
    else:
        liftf = make_lift_function_3d(config)
        domainf = make_domain_function_3d(config, diameter)
        implicit_solver = make_implicit_solver(config)
        return liftf, domainf, implicit_solver


def sag_surface_raytrace(
    sag_function: ti.BoundSagFunction,
    lift_function: ti.LiftFunction,
    domain_function: ti.DomainFunction,
    implicit_solver: ImplicitSolver,
    P: BatchNDTensor,
    V: BatchNDTensor,
    tf_in: Tf,
    diameter: ScalarTensor,
    normalize: Bool[torch.Tensor, ""],
) -> tuple[
    BatchTensor, BatchNDTensor, MaskTensor, BatchNDTensor, BatchNDTensor, BatchTensor
]:
    """
    Generic raytracing for a sag surface.
    Used to implement surface kernels.
    """

    # Setup the implicit function
    one = torch.ones((), dtype=P.dtype, device=P.device)
    tau = diameter / 2
    nf = torch.where(normalize, tau, one)
    implicit_function = lift_function(sag_function, nf, tau)

    # Setup the local solver
    local_solver = partial(
        implicit_surface_local_raytrace,
        implicit_function=implicit_function,
        domain_function=domain_function,
        implicit_solver=implicit_solver,
    )

    # Perform raytrace
    return surface_raytrace(P, V, tf_in, local_solver)


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

    def __init__(
        self, dim: int, sag: ti.SagFunction, solver_config: ImplicitSolverConfig
    ):
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

    def example_params(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[
        ScalarTensor,
        Bool[torch.Tensor, ""],
        torch.Tensor,
    ]:
        return (
            torch.tensor(10.0, dtype=dtype, device=device),
            torch.tensor(False, dtype=torch.bool, device=device),
            self.sag.example_params(dtype, device),
        )


class SagSumSurfaceKernel(FunctionalKernel):
    """
    Generic 2D/3D sag surface kernel for a sum of two sag terms
    """

    inputs = {"P": BatchNDTensor, "V": BatchNDTensor, "tf_in": Tf}

    params = {
        "diameter": ScalarTensor,
        "normalize": Bool[torch.Tensor, ""],
        "params1": torch.Tensor,
        "params2": torch.Tensor,
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
        dim: int,
        sag1: ti.SagFunction,
        sag2: ti.SagFunction,
        solver_config: ImplicitSolverConfig,
    ):
        self.dim = dim
        self.sag1 = sag1
        self.sag2 = sag2
        self.solver_config = solver_config

    def apply(
        self,
        P: BatchNDTensor,
        V: BatchNDTensor,
        tf_in: Tf,
        diameter: ScalarTensor,
        normalize: Bool[torch.Tensor, ""],
        params1: torch.Tensor,
        params2: torch.Tensor,
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

        sag_sum = ti.sag_sum_2d if self.dim == 2 else ti.sag_sum_3d

        # Bind sag function parameters for the solver
        def S(points: BatchNDTensor, *, order: int) -> ti.SagResult:
            return sag_sum(
                points,
                sags=[
                    partial(self.sag1, params=params1),
                    partial(self.sag2, params=params2),
                ],
                order=order,
            )

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

    def example_params(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[
        ScalarTensor,
        Bool[torch.Tensor, ""],
        torch.Tensor,
        torch.Tensor,
    ]:
        return (
            torch.tensor(10.0, dtype=dtype, device=device),
            torch.tensor(False, dtype=torch.bool, device=device),
            self.sag1.example_params(dtype, device),
            self.sag2.example_params(dtype, device),
        )


class SagOuterExtentSurfaceKernel(FunctionalKernel):
    inputs = {"anchor": ScalarTensor}
    params = {
        "diameter": ScalarTensor,
        "normalize": Bool[torch.Tensor, ""],
        "params": torch.Tensor,
    }
    outputs = {"extent": ScalarTensor}

    def __init__(self, sag: ti.SagFunction):
        assert sag.dim == 2, (
            f"SagOuterExtentSurfaceKernel requires a 2D sag function, got dim={sag.dim}"
        )
        self.sag = sag

    def apply(
        self,
        anchor: ScalarTensor,
        diameter: ScalarTensor,
        normalize: Bool[torch.Tensor, ""],
        params: torch.Tensor,
    ) -> ScalarTensor:
        extent_unnormalized = self.sag(
            anchor * diameter / 2, params=params, order=0
        ).val
        extent_normalized = diameter / 2 * self.sag(anchor, params, order=0).val
        extent = torch.where(normalize, extent_normalized, extent_unnormalized)
        return extent

    def example_inputs(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[(ScalarTensor)]:
        return (torch.tensor(1.0, dtype=dtype, device=device),)

    def example_params(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[ScalarTensor, Bool[torch.Tensor, ""], torch.Tensor]:
        return (
            torch.tensor(10.0, dtype=dtype, device=device),
            torch.tensor(True, dtype=torch.bool, device=device),
            self.sag.example_params(dtype, device),
        )
