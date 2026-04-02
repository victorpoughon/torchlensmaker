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
from typing import Any, Callable, Sequence, TypeAlias, TypedDict

import torch
from jaxtyping import Bool, Float

from torchlensmaker.surfaces.sag_geometry import (
    lens_diameter_implicit_domain_2d,
    lens_diameter_implicit_domain_3d,
)
from torchlensmaker.types import (
    Batch2DTensor,
    BatchNDTensor,
    BatchTensor,
    MaskTensor,
    ScalarTensor,
    Tf,
)

from .implicit_solver import (
    DomainFunction,
    ImplicitSolver,
    implicit_solver_newton,
    implicit_surface_local_raytrace,
)
from .raytrace import surface_raytrace
from .sag_functions import (
    LiftFunction,
    SagFunction,
    sag_to_implicit_2d_euclid,
    sag_to_implicit_2d_raw,
    sag_to_implicit_3d_raw,
)

SagSolverConfig: TypeAlias = dict[str, Any]
"""
Static configuration for raytracing sag surfaces using
an implicit solver

Possible values:
    * implicit_solver: implicit solver algorithm, supported values: "newton"
    * num_iter: number of iterations of the solver
    * damping: damping factor in ]0, 1]
    * tol: absolute tolerance for the domain function
    * lift_function: "raw" or "euclid"
"""


def _make_implicit_solver(config: SagSolverConfig) -> ImplicitSolver:
    num_iter: int = config["num_iter"]
    damping: float = config["damping"]
    solver_name: str = config["implicit_solver"]
    if solver_name == "newton":
        return partial(implicit_solver_newton, num_iter=num_iter, damping=damping)
    else:
        raise ValueError(f"Unknown implicit solver: {solver_name!r}")


def sag_solver_config_2d(
    config: SagSolverConfig, diameter: ScalarTensor
) -> tuple[LiftFunction, DomainFunction, ImplicitSolver]:
    tol: float = config["tol"]
    liftf = {
        "raw": sag_to_implicit_2d_raw,
        "euclid": sag_to_implicit_2d_euclid,
    }[config["lift_function"]]

    domainf = partial(lens_diameter_implicit_domain_2d, diameter=diameter, tol=tol)
    implicit_solver = _make_implicit_solver(config)

    return liftf, domainf, implicit_solver


def sag_solver_config_3d(
    config: SagSolverConfig, diameter: ScalarTensor
) -> tuple[LiftFunction, DomainFunction, ImplicitSolver]:
    tol: float = config["tol"]

    liftf = {
        "raw": sag_to_implicit_3d_raw,
        # "euclid": sag_to_implicit_3d_euclid,
    }[config["lift_function"]]
    domainf = partial(lens_diameter_implicit_domain_3d, diameter=diameter, tol=tol)
    implicit_solver = _make_implicit_solver(config)

    return liftf, domainf, implicit_solver


def sag_solver_config(
    dim: int, config: SagSolverConfig, diameter: ScalarTensor
) -> tuple[LiftFunction, DomainFunction, ImplicitSolver]:
    """
    Configure a sag function from static parameters
    """

    if dim == 2:
        return sag_solver_config_2d(config, diameter)
    else:
        return sag_solver_config_3d(config, diameter)


def sag_surface_raytrace(
    sag_function: SagFunction,
    lift_function: LiftFunction,
    domain_function: DomainFunction,
    implicit_solver: ImplicitSolver,
    P: BatchNDTensor,
    V: BatchNDTensor,
    tf_in: Tf,
    diameter: ScalarTensor,
    normalize: Bool[torch.Tensor, ""],
) -> tuple[BatchTensor, BatchNDTensor, MaskTensor, BatchNDTensor, BatchNDTensor]:
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
