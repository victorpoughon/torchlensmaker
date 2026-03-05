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

from typing import Sequence, Optional, TypeAlias, Literal, Self, Any
from torchlensmaker.types import BatchTensor, ScalarTensor
from torchlensmaker.core.base_module import BaseModule
from torchlensmaker.core.tensor_manip import to_tensor, filter_optional_tensor
from torchlensmaker.surfaces.surface_disk import Disk

from torchlensmaker.optical_data import OpticalData

from .surface_propagator import SurfacePropagator


def linear_magnification(
    field_coordinates: BatchTensor, image_coordinates: BatchTensor
) -> tuple[BatchTensor, BatchTensor]:
    T, V = field_coordinates, image_coordinates

    # Fit linear magnification with least square and compute residuals
    mag = torch.sum(T * V) / torch.sum(T**2)
    residuals = V - mag * T

    return mag, residuals


class ImagePlane(BaseModule):
    """
    Linear magnification disk image plane

    Loss function with a target magnification:
        L = (target_magnification - current_magnification)**2 + sum(residuals**2)

    Without:
        L = sum(residuals**2)
    """

    def __init__(
        self,
        diameter: float | ScalarTensor,
        magnification: Optional[int | float | ScalarTensor] = None,
    ):
        super().__init__()
        self.propagator = SurfacePropagator(Disk(diameter))
        self.magnification = (
            to_tensor(magnification) if magnification is not None else None
        )

    def clone(self, **overrides: Any) -> Self:
        kwargs = dict(
            diameter=self.propagator.surface.diameter,
            magnification=self.magnification,
        )
        return type(self)(**kwargs | overrides)

    def forward(self, data: OpticalData) -> OpticalData:
        if data.rays.V.shape[0] == 0:
            return data

        # Collision detection
        rays_propagated, _, _ = self.propagator(data.rays, data.fk)

        # TODO 2D only for now
        if data.dim == 3:
            return data

        # Compute image surface coordinates here
        # To make this work with any surface, we would need a way to compute
        # surface coordinates for points on a surface, for any surface
        # For a plane it's easy though

        collision_points = rays_propagated.P
        rays_image = collision_points[:, 1]
        rays_object = rays_propagated.field

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

        # Note: image plane is transparent to rays,
        # we only add information to the rays (the image plane coordinates)

        return data.replace(
            rays=data.rays.replace(image=rays_image),
            loss=loss,
        )

    def sequential(self, data: OpticalData) -> OpticalData:
        return self(data)
