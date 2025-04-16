# This file is part of Torch Lens Maker
# Copyright (C) 2025 Victor Poughon
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

from torchlensmaker.surfaces.local_surface import LocalSurface
from torchlensmaker.core.geometry import unit_vector
from torchlensmaker.core.cylinder_collision import (
    rays_cylinder_collision,
    rays_rectangle_collision,
)

from torchlensmaker.core.collision_detection import (
    CollisionMethod,
    default_collision_method,
)


from typing import TypeAlias

Tensor: TypeAlias = torch.Tensor


class ImplicitSurface(LocalSurface):
    """
    Surface3D defined in implicit form: F(x,y,z) = 0
    """

    def __init__(
        self,
        collision_method: CollisionMethod = default_collision_method,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__(dtype=dtype)
        self.collision_method = collision_method

    def bcyl(self) -> Tensor:
        raise NotImplementedError

    def rmse(self, points: Tensor) -> float:
        N = sum(points.shape[:-1])
        return torch.sqrt(torch.sum(self.Fd(points) ** 2) / N).item()

    def local_collide(self, P: Tensor, V: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        assert P.dtype == V.dtype == self.dtype
        assert P.shape == V.shape

        N, dim = P.shape
        dtype = P.dtype

        with torch.no_grad():
            # Get bounding cylinder and compute ray-cylinder intersection
            xmin, xmax, tau = self.bcyl().unbind()
            if dim == 3:
                t1, t2, hit_mask = rays_cylinder_collision(P, V, xmin, xmax, tau)
            else:
                t1, t2, hit_mask = rays_rectangle_collision(P, V, xmin, xmax, -tau, tau)

            # Split rays into definitly not colliding, and possibly colliding ("maybe")
            P_maybe, V_maybe = P[hit_mask], V[hit_mask]
            tmin, tmax = t1[hit_mask], t2[hit_mask]

        # Run iterative collision method on possibly colliding rays
        # TODO if no "maybe rays", skip this call, because tmin tmax are empty when N=0
        t = self.collision_method(self, P_maybe, V_maybe, tmin, tmax, history=False).t

        # two kinds of non colliding rays at this points:
        # - non colliding from before bounding cylinder check
        # - non colliding after iterations complete, t will make a non colliding point

        local_points = P_maybe + t.unsqueeze(-1).expand_as(V_maybe) * V_maybe
        local_normals = self.normals(local_points)
        valid = self.contains(local_points)

        # final t: t made into total N shape
        hit_mask_indices = hit_mask.nonzero().squeeze(-1)
        final_t = torch.zeros((N,), dtype=t.dtype).index_put((hit_mask_indices,), t)

        default_normal = unit_vector(dim, dtype)
        final_normals = (
            default_normal.unsqueeze(0)
            .expand(N, dim)
            .index_put((hit_mask_indices,), local_normals)
        )

        # final mask: cylinder_hit_mask combined with contains(local_points) mask
        final_valid = torch.full((N,), False).index_put((hit_mask_indices,), valid)

        return final_t, final_normals, final_valid

    def normals(self, points: Tensor) -> Tensor:
        # To get the normals of an implicit surface,
        # normalize the gradient of the implicit function
        return nn.functional.normalize(self.Fd_grad(points), dim=-1)

    def Fd(self, points: Tensor) -> Tensor:
        "Calls f or F depending on the shape of points"
        return self.f(points) if points.shape[-1] == 2 else self.F(points)

    def Fd_grad(self, points: Tensor) -> Tensor:
        "Calls f_grad or F_grad depending on the shape of points"
        return self.f_grad(points) if points.shape[-1] == 2 else self.F_grad(points)

    def f(self, points: Tensor) -> Tensor:
        raise NotImplementedError

    def f_grad(self, points: Tensor) -> Tensor:
        raise NotImplementedError

    def F(self, points: Tensor) -> Tensor:
        """
        Implicit equation for the 3D shape: F(x,y,z) = 0

        Args:
            points: tensor of shape (N, 3) where columns are X, Y, Z coordinates and N is the batch dimension

        Returns:
            F: value of F at the given points, tensor of shape (N,)
        """
        raise NotImplementedError

    def F_grad(self, points: Tensor) -> Tensor:
        """
        Gradient of F

        Args:
            points: tensor of shape (N, 3) where columns are X, Y, Z coordinates and N is the batch dimension

        Returns:
            F_grad: value of the gradient of F at the given points, tensor of shape (N, 3)
        """
        raise NotImplementedError
