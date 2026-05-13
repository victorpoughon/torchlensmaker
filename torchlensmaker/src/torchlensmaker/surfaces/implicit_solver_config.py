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
from typing import Any, TypeAlias

import torchimplicit as ti
from torchimplicit.lift_functions import (
    sag_to_implicit_2d_euclid,
    sag_to_implicit_2d_raw,
    sag_to_implicit_2d_taylor,
    sag_to_implicit_2d_taylor_squared,
    sag_to_implicit_3d_raw,
)

from torchlensmaker.surfaces.sag_geometry import (
    lens_diameter_implicit_domain_2d,
    lens_diameter_implicit_domain_3d,
)
from torchlensmaker.types import (
    ScalarTensor,
)

from .implicit_solver import (
    DomainFunction,
    ImplicitSolver,
    implicit_solver_newton,
    implicit_solver_newton2,
)

ImplicitSolverConfig: TypeAlias = dict[str, Any]
"""
Static configuration for raytracing sag surfaces using
an implicit solver

Possible values:
    * implicit_solver: implicit solver algorithm, supported values: "newton", "newton2"
    * num_iter: number of iterations of the solver
    * damping: damping factor in ]0, 1]
    * tol: absolute tolerance for the domain function
    * lift_function: "raw" or "euclid"
    * init: how to initialize t before Newton iterations, float or "closest"
    * clamp_positive: if True, clamp t >= 0 after each Newton update step
"""


def make_implicit_solver(config: ImplicitSolverConfig) -> ImplicitSolver:
    num_iter: int = config["num_iter"]
    damping: float = config["damping"]
    init: float | str = config["init"]
    clamp_positive: bool = config["clamp_positive"]
    solver_name: str = config["implicit_solver"]
    if solver_name == "newton":
        return partial(
            implicit_solver_newton,
            num_iter=num_iter,
            damping=damping,
            init=init,
            clamp_positive=clamp_positive,
        )
    elif solver_name == "newton2":
        return partial(
            implicit_solver_newton2,
            num_iter=num_iter,
            damping=damping,
            init=init,
            clamp_positive=clamp_positive,
        )
    else:
        raise ValueError(f"Unknown implicit solver: {solver_name!r}")


def make_lift_function_2d(config: ImplicitSolverConfig) -> ti.LiftFunction:
    lift_name: str = config["lift_function"]
    options = {
        "raw": sag_to_implicit_2d_raw,
        "abs": sag_to_implicit_2d_euclid,
        "taylor": sag_to_implicit_2d_taylor,
        "taylor_squared": sag_to_implicit_2d_taylor_squared,
        "euclid": sag_to_implicit_2d_euclid,
    }
    if lift_name not in options:
        raise ValueError(f"Unknown 2D lift function: {lift_name!r}")
    return options[lift_name]


def make_lift_function_3d(config: ImplicitSolverConfig) -> ti.LiftFunction:
    lift_name: str = config["lift_function"]
    options = {
        "raw": sag_to_implicit_3d_raw,
        # "euclid": sag_to_implicit_3d_euclid,
    }
    if lift_name not in options:
        raise ValueError(f"Unknown 3D lift function: {lift_name!r}")
    return options[lift_name]


def make_domain_function_2d(
    config: ImplicitSolverConfig, diameter: ScalarTensor
) -> DomainFunction:
    tol: float = config["tol"]
    return partial(lens_diameter_implicit_domain_2d, diameter=diameter, tol=tol)


def make_domain_function_3d(
    config: ImplicitSolverConfig, diameter: ScalarTensor
) -> DomainFunction:
    tol: float = config["tol"]
    return partial(lens_diameter_implicit_domain_3d, diameter=diameter, tol=tol)


def implicit_solver_config(dim: int, config: ImplicitSolverConfig) -> ImplicitSolver:
    """
    Configure implicit solver parameters from static parameters
    """

    if dim == 2:
        implicit_solver = make_implicit_solver(config)
        return implicit_solver
    else:
        implicit_solver = make_implicit_solver(config)
        return implicit_solver
