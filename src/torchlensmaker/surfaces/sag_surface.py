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
import torch
import torch.nn as nn

from torchlensmaker.surfaces.implicit_surface import ImplicitSurface
from torchlensmaker.core.geometry import within_radius
from torchlensmaker.core.collision_detection import (
    CollisionMethod,
    default_collision_method,
)
from torchlensmaker.core.sag_functions import SagFunction


from typing import TypeAlias, Optional, Any

Tensor: TypeAlias = torch.Tensor


class SagSurface(ImplicitSurface):
    """
    Axially symmetric implicit surface defined by a sag function.

    A sag function g(r) is a one dimensional real valued function that describes
    a surface x coordinate in an arbitrary meridional plane (x,r) as a function
    of the distance to the principal axis: x = g(r).

    Sag function classes provide the sag function and its gradient in
    both 2 and 3 dimensions. This class then uses it to create the implicit
    function F representing the corresponding implicit surface.

    The sag function is assumed defined only on the domain (- diameter/2 ;
    diameter / 2) in the meridional plane. Outside of this domain, a fallback
    function is used (implemented by DiameterBandSurfaceSq).
    """

    def __init__(
        self,
        diameter: float,
        sag_function: SagFunction,
        collision_method: CollisionMethod = default_collision_method,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__(collision_method=collision_method, dtype=dtype)
        self.diameter = diameter
        self.sag_function = sag_function

    def mask_function(self, points: Tensor) -> Tensor:
        return within_radius(self.diameter / 2, points)

    def parameters(self) -> dict[str, nn.Parameter]:
        return self.sag_function.parameters()

    # TODO remove?
    def bounding_radius(self) -> float:
        """
        Any point on the surface has a distance to the center that is less
        than (or equal) to the bounding radius
        """
        return math.sqrt((self.diameter / 2) ** 2 + self.extent_x() ** 2)

    def tau(self) -> Tensor:
        "Half-diameter and normalization factor"
        return torch.as_tensor(self.diameter / 2, dtype=self.dtype)

    def f(self, points: Tensor) -> Tensor:
        "points are assumed to be within the bcyl domain"
        assert points.shape[-1] == 2
        x, r = points.unbind(-1)
        return self.sag_function.g(r, self.tau()) - x

    def f_grad(self, points: Tensor) -> Tensor:
        assert points.shape[-1] == 2
        x, r = points.unbind(-1)
        return torch.stack(
            (-torch.ones_like(x), self.sag_function.g_grad(r, self.tau())), dim=-1
        )

    def F(self, points: Tensor) -> Tensor:
        assert points.shape[-1] == 3
        x, y, z = points.unbind(-1)
        return self.sag_function.G(y, z, self.tau()) - x

    def F_grad(self, points: Tensor) -> Tensor:
        assert points.shape[-1] == 3
        x, y, z = points.unbind(-1)
        grad_y, grad_z = self.sag_function.G_grad(y, z, self.tau())
        return torch.stack((-torch.ones_like(x), grad_y, grad_z), dim=-1)

    def extent_x(self) -> Tensor:
        r = torch.as_tensor(self.diameter / 2, dtype=self.dtype)
        return self.sag_function.g(r, tau=r)

    def bcyl(self) -> Tensor:
        """Bounding cylinder
        Returns a tensor of shape (3,) where entries are [xmin, xmax, radius]
        """

        tau = torch.tensor(self.diameter / 2, dtype=self.dtype)
        return torch.cat(
            (
                self.sag_function.bounds(tau),
                tau.unsqueeze(0),
            ),
            dim=0,
        )

    def contains(self, points: Tensor, tol: Optional[float] = None) -> Tensor:
        if tol is None:
            tol = {torch.float32: 1e-4, torch.float64: 1e-7}[self.dtype]

        N, dim = points.shape

        # Check points are within the diameter
        r = (
            torch.abs(points[:, 1])
            if dim == 2
            else torch.sqrt(points[:, 1] ** 2 + points[:, 2] ** 2)
        )
        within_diameter = r <= self.diameter

        tau = self.tau()
        zeros1d = torch.zeros_like(points[:, 1])
        zeros2d = torch.zeros_like(r)

        # If within diameter, check the sag equation x = g(r)
        if dim == 2:
            safe_input = torch.where(within_diameter, r, zeros2d)
            sagG = self.sag_function.g(safe_input, tau)
            G = torch.where(within_diameter, sagG, zeros2d)
        else:
            safe_input_y = torch.where(within_diameter, points[:, 1], zeros1d)
            safe_input_z = torch.where(within_diameter, points[:, 2], zeros1d)
            sagG = self.sag_function.G(safe_input_y, safe_input_z, tau)
            G = torch.where(within_diameter, sagG, zeros2d)

        within_tol = torch.abs(G - points[:, 0]) < tol
        return torch.logical_and(within_diameter, within_tol)

    def samples2D_full(self, N: int, epsilon: float) -> torch.Tensor:
        start = -(1 - epsilon) * self.diameter / 2
        end = (1 - epsilon) * self.diameter / 2
        r = torch.linspace(start, end, N, dtype=self.dtype)
        x = self.sag_function.g(r, self.tau())
        return torch.stack((x, r), dim=-1)

    def samples2D_half(self, N: int, epsilon: float) -> torch.Tensor:
        start = 0.0
        end = (1 - epsilon) * self.diameter / 2
        r = torch.linspace(start, end, N, dtype=self.dtype)
        x = self.sag_function.g(r, self.tau())
        return torch.stack((x, r), dim=-1)

    def to_dict(self, dim: int) -> dict[str, Any]:
        return {
            "type": "surface-sag",
            "diameter": self.diameter,
            "sag-function": self.sag_function.to_dict(dim),
            "bcyl": self.bcyl().tolist(),
        }
