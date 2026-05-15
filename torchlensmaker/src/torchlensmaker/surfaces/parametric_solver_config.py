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

from .parametric_solver import (
    THETA_INIT_FUNCTIONS,
    ParametricDomainFunction,
    ParametricFunction,
    ParametricSolver,
    ThetaInitFunction,
    init_theta_constant,
    parametric_residual_domain,
    parametric_solver_newton,
    parametric_solver_newton2,
)

ParametricSolverConfig: TypeAlias = dict[str, Any]
"""
Static configuration for raytracing parametric surfaces.

Possible values:
    * parametric_solver: solver algorithm, supported values: "newton", "newton2"
    * num_iter: number of Newton iterations
    * damping: damping factor in ]0, 1]
    * tol: absolute tolerance on residual ||P + tV - S(uv)|| for the domain function
    * init: how to initialize t before Newton iterations, float or "closest"
    * clamp_positive: if True, clamp t >= 0 after each Newton update step
    * singular_check: if True, raise LinAlgError when the Jacobian is singular
"""


def make_init_function(init: float | str) -> ThetaInitFunction:
    if isinstance(init, str) and init in THETA_INIT_FUNCTIONS:
        return THETA_INIT_FUNCTIONS[init]
    return partial(init_theta_constant, t=float(init))


def make_parametric_solver(config: ParametricSolverConfig) -> ParametricSolver:
    num_iter: int = config["num_iter"]
    damping: float = config["damping"]
    init_fn: ThetaInitFunction = make_init_function(config["init"])
    clamp_positive: bool = config["clamp_positive"]
    singular_check: bool = config["singular_check"]
    solver_name: str = config["parametric_solver"]

    if solver_name == "newton":
        return partial(
            parametric_solver_newton,
            num_iter=num_iter,
            damping=damping,
            init_fn=init_fn,
            clamp_positive=clamp_positive,
            singular_check=singular_check,
        )
    elif solver_name == "newton2":
        return partial(
            parametric_solver_newton2,
            num_iter=num_iter,
            damping=damping,
            init_fn=init_fn,
            clamp_positive=clamp_positive,
            singular_check=singular_check,
        )
    else:
        raise ValueError(f"Unknown parametric solver: {solver_name!r}")


def make_domain_function(
    config: ParametricSolverConfig, parametric_function: ParametricFunction
) -> ParametricDomainFunction:
    tol: float = config["tol"]
    return partial(parametric_residual_domain, parametric_function=parametric_function, tol=tol)
