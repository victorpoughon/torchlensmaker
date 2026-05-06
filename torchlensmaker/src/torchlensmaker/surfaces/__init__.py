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

from torchimplicit.lift_functions import (
    LiftFunction,
    sag_to_implicit_2d_abs,
    sag_to_implicit_2d_euclid,
    sag_to_implicit_2d_euclid_squared,
    sag_to_implicit_2d_raw,
    sag_to_implicit_2d_squared_blend,
    sag_to_implicit_2d_taylor,
    sag_to_implicit_2d_taylor_squared,
    sag_to_implicit_3d_raw,
)
from torchimplicit.sag_functions import (
    BoundSagFunction,
    aspheric_sag_2d,
    aspheric_sag_3d,
    conical_sag_2d,
    conical_sag_3d,
    parabolic_sag_2d,
    parabolic_sag_3d,
    sag_sum_2d,
    sag_sum_3d,
    spherical_sag_2d,
    spherical_sag_3d,
    xypolynomial_sag_3d,
)

from .implicit_solver import (
    DomainFunction,
    ImplicitSolver,
    implicit_solver_newton,
    implicit_solver_newton2,
)
from .surface_anchor import KinematicSurface
from .surface_asphere import Asphere
from .surface_conic import Conic
from .surface_disk import Disk
from .surface_element import SurfaceElement, SurfaceElementOutput
from .surface_implicit import ImplicitDisk, Sphere
from .surface_parabola import Parabola
from .surface_plane import Plane
from .surface_point import PointSurface
from .surface_sphere_by_curvature import SphereByCurvature
from .surface_sphere_by_radius import SphereByRadius
from .surface_square import Square
from .surface_xypolynomial import XYPolynomial

__all__ = [
    # Surface classes
    "Asphere",
    "Conic",
    "Disk",
    "ImplicitDisk",
    "KinematicSurface",
    "Parabola",
    "Plane",
    "PointSurface",
    "SphereByCurvature",
    "Sphere",
    "SphereByRadius",
    "Square",
    "SurfaceElement",
    "SurfaceElementOutput",
    "XYPolynomial",
    # Sag functions
    "aspheric_sag_2d",
    "aspheric_sag_3d",
    "conical_sag_2d",
    "conical_sag_3d",
    "parabolic_sag_2d",
    "parabolic_sag_3d",
    "sag_sum_2d",
    "sag_sum_3d",
    "sag_to_implicit_2d_euclid",
    "sag_to_implicit_2d_raw",
    "sag_to_implicit_2d_abs",
    "sag_to_implicit_2d_taylor",
    "sag_to_implicit_2d_taylor_squared",
    "sag_to_implicit_2d_euclid_squared",
    "sag_to_implicit_2d_squared_blend",
    "sag_to_implicit_3d_raw",
    "spherical_sag_2d",
    "spherical_sag_3d",
    "xypolynomial_sag_3d",
    # Sag type aliases
    "LiftFunction",
    "BoundSagFunction",
    # Solver
    "implicit_solver_newton",
    "implicit_solver_newton2",
    # Solver type aliases
    "DomainFunction",
    "ImplicitSolver",
]
