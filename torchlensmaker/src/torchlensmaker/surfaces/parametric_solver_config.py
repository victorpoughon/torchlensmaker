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
    ParametricDomainFunction,
    ParametricFunction,
    ParametricSolver,
    parametric_residual_domain,
    parametric_solver_newton,
)

ParametricSolverConfig: TypeAlias = dict[str, Any]
"""
Static configuration for raytracing parametric surfaces.

Possible values:
    * parametric_solver: solver algorithm, supported values: "newton"
    * num_iter: number of Newton iterations
    * damping: damping factor in ]0, 1]
    * tol: absolute tolerance on residual ||P + tV - S(uv)|| for the domain function
    * init: how to initialize t before Newton iterations, float or "closest"
    * clamp_positive: if True, clamp t >= 0 after each Newton update step
"""


def make_parametric_solver(config: ParametricSolverConfig) -> ParametricSolver:
    num_iter: int = config["num_iter"]
    damping: float = config["damping"]
    init: float | str = config["init"]
    clamp_positive: bool = config["clamp_positive"]
    solver_name: str = config["parametric_solver"]

    if solver_name == "newton":
        return partial(
            parametric_solver_newton,
            num_iter=num_iter,
            damping=damping,
            init=init,
            clamp_positive=clamp_positive,
        )
    else:
        raise ValueError(f"Unknown parametric solver: {solver_name!r}")


def make_domain_function(
    config: ParametricSolverConfig, parametric_function: ParametricFunction
) -> ParametricDomainFunction:
    tol: float = config["tol"]
    return partial(parametric_residual_domain, parametric_function=parametric_function, tol=tol)
