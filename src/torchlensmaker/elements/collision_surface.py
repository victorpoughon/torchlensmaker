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

from torchlensmaker.types import Tf, Tf2D, Tf3D

from torchlensmaker.core.tensor_manip import to_tensor, filter_optional_tensor
from torchlensmaker.kinematics.homogeneous_geometry import (
    kinematic_chain_extend,
    kinematic_chain_append,
    hom_translate,
    hom_identity,
    hom_scale,
)
from torchlensmaker.surfaces.local_surface import LocalSurface
from torchlensmaker.core.intersect import intersect

from torchlensmaker.optical_data import OpticalData

Tensor = torch.Tensor


class CollisionSurface(nn.Module):
    """
    Surface rays collision detection

    CollisionSurface implements collision detection between a bundle of rays
    (represented by an OpticalData object) and a surface. It also positions the
    surface according to anchors and scale options.

    Returns:
        t: full frame collision distances
        normals: full frame normals at the collision points
        valid: mask indicating which input rays collide with the surface
        chain: kinematic chain for the next element
    """

    def __init__(
        self,
        surface: LocalSurface,
        scale: float = 1.0,
        anchors: tuple[str, str] = ("origin", "origin"),
    ):
        super().__init__()
        self.surface = surface
        self.scale = scale
        self.anchors = anchors

        # If surface has parameters, register them
        for name, p in surface.parameters().items():
            self.register_parameter(name, p)

    def kinematic_transform(self, dim: int, dtype: torch.dtype) -> Tf:
        "Additional transform that applies to the next element"

        assert dtype == self.surface.dtype
        device = torch.device("cpu")  # TODO gpu support

        # Subtract first anchor, add second anchor

        # TODO surface.extent() is undefined for some surfaces (XYPolynomial)
        # so avoid computing it if we don't need it

        if self.anchors == ("extent", "extent"):
            t0 = hom_translate(-self.scale * self.surface.extent(dim))
            t1 = hom_translate(self.scale * self.surface.extent(dim))
            return kinematic_chain_append(t0, t1)
        elif self.anchors == ("extent", "origin"):
            t0 = hom_translate(-self.scale * self.surface.extent(dim))
            return t0
        elif self.anchors == ("origin", "extent"):
            t1 = hom_translate(self.scale * self.surface.extent(dim))
            return t1
        else:
            return hom_identity(dim, dtype, device)

    def surface_transform(self, dim: int, dtype: torch.dtype) -> Tf:
        "Additional transform that applies to the surface"

        assert dtype == self.surface.dtype
        device = torch.device("cpu")  # TODO gpu support

        tf_scale = hom_scale(
            dim, torch.as_tensor(self.scale, dtype=dtype, device=device)
        )

        if self.anchors[0] == "extent":
            tf_anchor = hom_translate(-self.scale * self.surface.extent(dim))
            return kinematic_chain_append(tf_anchor, tf_scale)
        else:
            return tf_scale

    def forward(self, inputs: OpticalData) -> tuple[Tensor, Tensor, Tensor, Tf]:
        dim, dtype = inputs.dim, inputs.dtype

        # Collision detection with the surface
        tf_surface = self.surface_transform(dim, dtype)
        fk_surface = kinematic_chain_append(inputs.fk, tf_surface)

        t, collision_normals, collision_valid = intersect(
            self.surface,
            inputs.P,
            inputs.V,
            fk_surface.direct,
            fk_surface.inverse,
        )

        tf_kinematic = self.kinematic_transform(dim, dtype)
        fk_next = kinematic_chain_append(inputs.fk, tf_kinematic)

        return t, collision_normals, collision_valid, fk_next
