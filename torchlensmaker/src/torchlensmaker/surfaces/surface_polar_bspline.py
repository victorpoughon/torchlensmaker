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

import math
from typing import Any, Self

import tlmviewer as tlmv
import torch

from torchlensmaker.core.tensor_manip import init_param
from torchlensmaker.raytracing.parametric_solver_config import (
    InitClosest,
    ParametricSolverConfig,
)
from torchlensmaker.surfaces.surface_bspline import BSplineSurfaceKernel
from torchlensmaker.types import (
    BatchTensor,
    Tf,
)

from .surface_element import SurfaceElement, SurfaceRecord


class PolarBSplineSurface(SurfaceElement):
    """
    Topologically-spherical B-spline surface with two pole points.

    Periodic in V (longitude), clamped in U (pole-to-pole). The poles are single
    trainable points; the surface passes exactly through them because all L control
    points in the first and last rows are broadcast from a single (3,) vector.

    Parameters
    ----------
    body_points : (K, L, 3) tensor
        Interior control point grid.
    north_pole : (3,) tensor
        Pole at u=0.
    south_pole : (3,) tensor
        Pole at u=1.

    Note: the surface normal is degenerate at the poles (dS/dv = 0 there). Rays
    that converge exactly at the poles will produce NaN normals. In practice this
    is not an issue as long as the optical beam avoids the pole region.
    """

    default_config: ParametricSolverConfig = {
        "parametric_solver": "newton",
        "num_iter": 10,
        "damping": 1.0,
        "tol": 1e-4,
        "init": InitClosest(),
        "clamp_positive": False,
        "singular_check": False,
        "periodic_uv": (False, True),
        "u_epsilon": 0.0,
        "v_epsilon": 0.0,
    }

    def __init__(
        self,
        body_points: torch.Tensor | torch.nn.Parameter,
        north_pole: torch.Tensor | torch.nn.Parameter,
        south_pole: torch.Tensor | torch.nn.Parameter,
        *,
        trainable: bool = False,
        solver_config: dict[str, Any] = {},
    ):
        super().__init__()

        self.solver_config = ParametricSolverConfig(
            **self.default_config | solver_config
        )

        self.degree = (2, 2)
        self.periodic = (False, True)
        self.clamped = (True, False)

        self.body_points = init_param(self, "body_points", body_points, trainable)
        self.north_pole = init_param(self, "north_pole", north_pole, trainable)
        self.south_pole = init_param(self, "south_pole", south_pole, trainable)

        self.func3d = BSplineSurfaceKernel(
            self.degree, self.periodic, self.clamped, self.solver_config
        )

    @classmethod
    def from_sphere(
        cls,
        radius: float,
        K: int,
        L: int,
        *,
        trainable: bool = False,
        solver_config: dict[str, Any] = {},
    ) -> Self:
        """
        Initialize with K*L body control points placed on a sphere of the given radius.

        Polar axis is along y. The equatorial belt lies in the xz plane.
        K body rows are evenly distributed between the poles in latitude.
        L columns are evenly distributed around the longitude.

        The modeled B-spline surface approximates a sphere but does not
        interpolate it exactly.
        """
        angles = torch.linspace(0, 2 * math.pi * (1 - 1 / L), L)
        us = torch.linspace(1.0 / (K + 1), K / (K + 1), K)
        thetas = us * math.pi  # polar angle in [0, pi] from the -y axis

        ring_radii = radius * torch.sin(thetas)
        y_values = -radius * torch.cos(thetas)

        body_points = torch.zeros(K, L, 3)
        body_points[:, :, 0] = ring_radii[:, None] * torch.cos(angles)[None, :]
        body_points[:, :, 1] = y_values[:, None].expand(K, L)
        body_points[:, :, 2] = ring_radii[:, None] * torch.sin(angles)[None, :]

        north_pole = torch.tensor([0.0, -radius, 0.0])
        south_pole = torch.tensor([0.0, radius, 0.0])

        return cls(
            body_points,
            north_pole,
            south_pole,
            trainable=trainable,
            solver_config=solver_config,
        )

    def _full_control_points(self) -> torch.Tensor:
        _K, L, _ = self.body_points.shape
        north = self.north_pole.expand(1, L, 3)
        south = self.south_pole.expand(1, L, 3)
        return torch.cat([north, self.body_points, south], dim=0)

    def clone(self, **overrides: Any) -> Self:
        kwargs: dict[str, Any] = dict(
            body_points=self.body_points,
            north_pole=self.north_pole,
            south_pole=self.south_pole,
            trainable=self.body_points.requires_grad,
            solver_config=self.solver_config,
        )
        return type(self)(**kwargs | overrides)

    def __repr__(self) -> str:
        return (
            f"{self._get_name()}("
            f"body_points={self.body_points.tolist()}, "
            f"north_pole={self.north_pole.tolist()}, "
            f"south_pole={self.south_pole.tolist()})"
        )

    def reverse(self) -> Self:
        return self.clone()

    def forward(self, P: BatchTensor, V: BatchTensor, tf: Tf) -> SurfaceRecord:
        t, normal, valid, points_local, points_global, rsm = self.func3d.apply(
            P, V, tf, self._full_control_points()
        )

        return SurfaceRecord(
            t, normal, valid, points_local, points_global, rsm, tf.clone(), tf.clone()
        )

    def render(self, matrix: torch.Tensor) -> tlmv.SurfaceBSpline:
        return tlmv.SurfaceBSpline(
            points=self._full_control_points().tolist(),
            degree=self.degree,
            periodic=self.periodic,
            clamped=self.clamped,
            samples=(100, 100),
            matrix=matrix.tolist(),
        )
