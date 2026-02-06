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

import torch
import torch.nn as nn
import math

from torchlensmaker.surfaces.local_surface import LocalSurface
from torchlensmaker.core.sphere_sampling import (
    sphere_samples_angular,
    sphere_samples_linear,
)
from torchlensmaker.core.geometry import unit_vector, within_radius
from torchlensmaker.core.collision_detection import init_closest_origin


from typing import Optional, Any

Tensor = torch.Tensor


class SphereR(LocalSurface):
    """
    A section of a sphere, parameterized by signed radius.

    This parameterization is useful to represent high curvature sections
    including a complete half-sphere. However it's poorly suited to represent
    low curvature sections that are closer to a planar surface.

    In 2D, this surface is an arc of circle.
    In 3D, this surface is a section of a sphere (wikipedia call it a "spherical cap")
    """

    def __init__(
        self,
        diameter: float,
        R: int | float | nn.Parameter | None = None,
        C: int | float | nn.Parameter | None = None,
        dtype: torch.dtype | None = None,
    ):
        if dtype is None:
            dtype = torch.get_default_dtype()

        super().__init__(dtype=dtype)
        self.diameter = diameter

        if (R is not None and C is not None) or (R is None and C is None):
            raise RuntimeError(
                "SphereR must be initialized with exactly one of R (radius) or C (curvature)."
            )

        self.R: torch.Tensor
        if C is None:
            if torch.abs(torch.as_tensor(R)) < diameter / 2:
                raise RuntimeError(
                    f"Sphere radius (R={R}) must be at least half the surface diameter (D={diameter})"
                )

            if isinstance(R, nn.Parameter):
                self.R = R
            else:
                self.R = torch.as_tensor(R, dtype=self.dtype)
        else:
            if isinstance(C, nn.Parameter):
                self.R = nn.Parameter(torch.tensor(1.0 / C.item(), dtype=self.dtype))
            else:
                self.R = torch.as_tensor(1.0 / C, dtype=self.dtype)

        assert self.R.dim() == 0
        assert self.R.dtype == self.dtype

    def parameters(self) -> dict[str, nn.Parameter]:
        if isinstance(self.R, nn.Parameter):
            return {"R": self.R}
        else:
            return {}

    def radius(self) -> float:
        return self.R.item()

    def extent_x(self) -> Tensor:
        r = self.diameter / 2
        K = 1 / self.R
        return torch.div(K * r**2, 1 + torch.sqrt(1 - (r * K) ** 2))

    def bounding_radius(self) -> float:
        return math.sqrt((self.diameter / 2) ** 2 + self.extent_x() ** 2)

    def center(self, dim: int) -> Tensor:
        if dim == 2:
            return torch.tensor([self.R, 0.0], dtype=self.dtype)
        else:
            return torch.tensor([self.R, 0.0, 0.0], dtype=self.dtype)

    def normals(self, points: Tensor) -> Tensor:
        batch, dim, dtype = points.shape[:-1], points.shape[-1], self.dtype

        # The normal is the vector from the center to the points
        center = self.center(dim)
        normals = torch.nn.functional.normalize(points - center, dim=-1)

        # We need a default value for the case where point == center, to avoid div by zero
        unit = unit_vector(dim, dtype)
        normal_at_origin = torch.tile(unit, ((*batch, 1)))

        return torch.where(
            torch.all(torch.isclose(center, points), dim=-1)
            .unsqueeze(-1)
            .expand_as(normals),
            normal_at_origin,
            normals,
        )

    def contains(self, points: Tensor, tol: Optional[float] = None) -> Tensor:
        if tol is None:
            tol = {torch.float32: 1e-3, torch.float64: 1e-7}[self.dtype]

        center = self.center(dim=points.shape[-1])
        within_outline = within_radius(self.diameter / 2 + tol, points)
        on_sphere = (
            torch.abs(
                torch.linalg.vector_norm(points - center, dim=-1) - torch.abs(self.R)
            )
            <= tol
        )
        within_extent = torch.abs(points[:, 0]) <= torch.abs(self.extent_x()) + tol

        return torch.all(
            torch.stack((within_outline, on_sphere, within_extent), dim=-1), dim=-1
        )

    def local_collide(self, P: Tensor, V: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        N, D = P.shape

        # For numerical stability, it's best if P is close to the origin
        # Bring rays origins as close as possible before solving
        init_t = init_closest_origin(self, P, V)
        P = P + init_t.unsqueeze(-1).expand_as(V) * V

        # Sphere-ray collision is a second order polynomial
        center = self.center(dim=D)
        A = torch.sum(V**2, dim=1)
        B = 2 * torch.sum(V * (P - center), dim=1)
        C = torch.sum((P - center) ** 2, dim=1) - self.R**2
        assert A.shape == B.shape == C.shape == (N,)

        delta = B**2 - 4 * A * C
        safe_delta = torch.clamp(delta, min=0.0)
        assert delta.shape == (N,), delta.shape
        assert safe_delta.shape == (N,), safe_delta.shape

        # tensor of shape (N, 2) with both roots
        # safe meaning that if the root is undefined the value is zero instead
        safe_roots = torch.stack(
            (
                (-B + torch.sqrt(safe_delta)) / (2 * A),
                (-B - torch.sqrt(safe_delta)) / (2 * A),
            ),
            dim=1,
        )
        assert safe_roots.shape == (N, 2)

        # mask of shape (N, 2) indicating if each root is inside the outline
        root_inside = torch.stack(
            (
                self.contains(P + safe_roots[:, 0].unsqueeze(1).expand_as(V) * V),
                self.contains(P + safe_roots[:, 1].unsqueeze(1).expand_as(V) * V),
            ),
            dim=1,
        )
        assert root_inside.shape == (N, 2)

        # number of valid roots
        number_of_valid_roots = torch.sum(root_inside, dim=1)
        assert number_of_valid_roots.shape == (N,)

        # index of the first valid root
        _, index_first_valid = torch.max(root_inside, dim=1)
        assert index_first_valid.shape == (N,)

        # index of the root closest to zero
        _, index_closest = torch.min(torch.abs(safe_roots), dim=1)
        assert index_closest.shape == (N,)

        # delta < 0 => no collision
        # delta >=0 => two roots (which maybe equal)
        #  - if both are outside the outline => no collision
        #  - if only one is inside the outline => one collision
        #  - if both are inside the outline => return the root closest to zero (i.e. the ray origin)

        # TODO refactor don't rely on contains at all

        default_t = torch.zeros(N, dtype=self.dtype)
        arange = torch.arange(N)
        t = torch.where(
            delta < 0,
            default_t,
            torch.where(
                number_of_valid_roots == 0,
                default_t,
                torch.where(
                    number_of_valid_roots == 1,
                    safe_roots[arange, index_first_valid],
                    torch.where(
                        number_of_valid_roots == 2,
                        safe_roots[arange, index_closest],
                        default_t,
                    ),
                ),
            ),
        )

        local_points = P + t.unsqueeze(1).expand_as(V) * V
        local_normals = self.normals(local_points)
        valid = number_of_valid_roots > 0

        return init_t + t, local_normals, valid

    def to_dict(self, dim: int) -> dict[str, Any]:
        return {
            "type": "surface-sphere-r",
            "diameter": self.diameter,
            "R": self.R.item(),
        }
