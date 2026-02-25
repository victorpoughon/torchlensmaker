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

from typing import Sequence, Optional, TypeAlias, Literal, Self

from torchlensmaker.core.tensor_manip import to_tensor, filter_optional_tensor
from torchlensmaker.kinematics.homogeneous_geometry import (
    kinematic_chain_extend,
    kinematic_chain_append,
    hom_translate,
    hom_identity,
    HomMatrix,
    hom_scale,
)
from torchlensmaker.surfaces.local_surface import LocalSurface
from torchlensmaker.surfaces.plane import CircularPlane
from torchlensmaker.materials.get_material_model import (
    MaterialModel,
    get_material_model,
)
from torchlensmaker.physics.physics import (
    refraction,
    reflection,
)
from torchlensmaker.optical_data import OpticalData
from torchlensmaker.elements.sequential import SequentialElement
from torchlensmaker.elements.collision_surface import CollisionSurface

Tensor = torch.Tensor


def linear_magnification(
    object_coordinates: Tensor, image_coordinates: Tensor
) -> tuple[Tensor, Tensor]:
    T, V = object_coordinates, image_coordinates

    # Fit linear magnification with least square and compute residuals
    mag = torch.sum(T * V) / torch.sum(T**2)
    residuals = V - mag * T

    return mag, residuals


class ImagePlane(SequentialElement):
    """
    Linear magnification circular image plane

    Loss function with a target magnification:
        L = (target_magnification - current_magnification)**2 + sum(residuals**2)

    Without:
        L = sum(residuals**2)
    """

    def __init__(
        self,
        diameter: float,
        magnification: Optional[int | float | Tensor] = None,
    ):
        super().__init__()
        self.collision_surface = CollisionSurface(CircularPlane(diameter))
        self.magnification = (
            to_tensor(magnification) if magnification is not None else None
        )

    def forward(self, data: OpticalData) -> OpticalData:
        if data.V.shape[0] == 0:
            return data

        # Collision detection
        t, _, valid_collision, _ = self.collision_surface(data)
        collision_points = data.P + t.unsqueeze(-1).expand_as(data.V) * data.V

        if data.rays_field is None:
            raise RuntimeError(
                "Missing object coordinates on rays (required to compute image magnification)"
            )

        # TODO 2D only for now
        if data.dim == 3:
            return data

        # Compute image surface coordinates here
        # To make this work with any surface, we would need a way to compute
        # surface coordinates for points on a surface, for any surface
        # For a plane it's easy though

        rays_image = collision_points[:, 1]
        rays_object = data.rays_field

        # Compute loss
        assert rays_object.shape == rays_image.shape, (
            rays_object.shape,
            rays_image.shape,
        )
        mag, res = linear_magnification(rays_object, rays_image)

        if self.magnification is not None:
            loss = (self.magnification - mag) ** 2 + torch.sum(torch.pow(res, 2))
        else:
            loss = torch.sum(torch.pow(res, 2))

        # what to do with rays after an image plane?

        return data.replace(
            rays_image=rays_image,  # used by spot_diagram
            loss=loss,
        )
