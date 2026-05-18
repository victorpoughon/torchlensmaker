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


from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, TypeAlias

from .parametric_solver import (
    ParametricDomainFunction,
    ParametricFunction,
    ParametricSolver,
    ThetaInitFunction,
    init_theta_closest,
    init_theta_constant,
    init_theta_grid_search,
    parametric_residual_domain,
    parametric_solver_newton,
    parametric_solver_newton2,
)


@dataclass
class InitClosest:
    """Initialize t to the closest approach to the origin, u and v to 0.5"""

    method: Literal["closest"] = "closest"


@dataclass
class InitConstant:
    """Initialize t to a constant value, u and v to 0.5"""

    t: float
    method: Literal["constant"] = "constant"


@dataclass
class InitGridSearch:
    """Initialize by grid search: evaluate S on a (t, u, v) grid and pick the minimum-distance point"""

    t_range: tuple[float, float]
    t_samples: int
    u_range: tuple[float, float] = (0.0, 1.0)
    u_samples: int = 5
    v_range: tuple[float, float] = (0.0, 1.0)
    v_samples: int = 5
    method: Literal["grid_search"] = "grid_search"


ThetaInit: TypeAlias = InitClosest | InitConstant | InitGridSearch

ParametricSolverConfig: TypeAlias = dict[str, Any]
"""
Static configuration for raytracing parametric surfaces.

Possible values:
    * parametric_solver: solver algorithm, supported values: "newton", "newton2"
    * num_iter: number of Newton iterations
    * damping: damping factor in ]0, 1]
    * tol: absolute tolerance on residual ||P + tV - S(uv)|| for the domain function
    * init: ThetaInit instance (InitClosest, InitConstant, or InitGridSearch)
    * clamp_positive: if True, clamp t >= 0 after each Newton update step
    * singular_check: if True, raise LinAlgError when the Jacobian is singular
    * periodic_uv: pair of bools; periodic dims are wrapped with remainder instead of clamped
    * u_epsilon: clamp u to [u_epsilon, 1 - u_epsilon] instead of [0, 1]
    * v_epsilon: clamp v to [v_epsilon, 1 - v_epsilon] instead of [0, 1]
"""


def make_init_function(init: ThetaInit) -> ThetaInitFunction:
    if isinstance(init, InitClosest):
        return init_theta_closest
    elif isinstance(init, InitConstant):
        return partial(init_theta_constant, t=init.t)
    elif isinstance(init, InitGridSearch):
        return partial(
            init_theta_grid_search,
            t_range=init.t_range,
            t_samples=init.t_samples,
            u_range=init.u_range,
            u_samples=init.u_samples,
            v_range=init.v_range,
            v_samples=init.v_samples,
        )
    else:
        raise ValueError(f"Unknown init: {init!r}")


def make_parametric_solver(config: ParametricSolverConfig) -> ParametricSolver:
    num_iter: int = config["num_iter"]
    damping: float = config["damping"]
    init_fn: ThetaInitFunction = make_init_function(config["init"])
    clamp_positive: bool = config["clamp_positive"]
    singular_check: bool = config["singular_check"]
    periodic_uv: tuple[bool, bool] = config["periodic_uv"]
    u_epsilon: float = config["u_epsilon"]
    v_epsilon: float = config["v_epsilon"]
    solver_name: str = config["parametric_solver"]

    if solver_name == "newton":
        return partial(
            parametric_solver_newton,
            num_iter=num_iter,
            damping=damping,
            init_fn=init_fn,
            clamp_positive=clamp_positive,
            singular_check=singular_check,
            periodic_uv=periodic_uv,
            u_epsilon=u_epsilon,
            v_epsilon=v_epsilon,
        )
    elif solver_name == "newton2":
        return partial(
            parametric_solver_newton2,
            num_iter=num_iter,
            damping=damping,
            init_fn=init_fn,
            clamp_positive=clamp_positive,
            singular_check=singular_check,
            periodic_uv=periodic_uv,
            u_epsilon=u_epsilon,
            v_epsilon=v_epsilon,
        )
    else:
        raise ValueError(f"Unknown parametric solver: {solver_name!r}")


def make_domain_function(
    config: ParametricSolverConfig, parametric_function: ParametricFunction
) -> ParametricDomainFunction:
    tol: float = config["tol"]
    return partial(
        parametric_residual_domain, parametric_function=parametric_function, tol=tol
    )
