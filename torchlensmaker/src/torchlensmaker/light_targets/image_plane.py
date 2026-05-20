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

from typing import Any, Optional, Self

import torch

from torchlensmaker.core.ray_bundle import RayBundle
from torchlensmaker.core.tensor_manip import to_tensor
from torchlensmaker.light_targets.light_target import LightTarget, LightTargetRecord
from torchlensmaker.sequential.optical_trace import OpticalTrace
from torchlensmaker.surfaces import Disk
from torchlensmaker.types import BatchTensor, ScalarTensor, Tf


def linear_magnification(
    field_coordinates: BatchTensor, image_coordinates: BatchTensor
) -> tuple[BatchTensor, BatchTensor]:
    T, V = field_coordinates, image_coordinates

    # Fit linear magnification with least square and compute residuals
    mag = torch.sum(T * V) / torch.sum(T**2)
    residuals = V - mag * T

    return mag, residuals


class ImagePlane(LightTarget):
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
        self.surface = Disk(diameter)
        self.magnification = (
            to_tensor(magnification) if magnification is not None else None
        )

    def clone(self, **overrides: Any) -> Self:
        kwargs: dict[str, Any] = dict(
            diameter=self.surface.diameter,
            magnification=self.magnification,
        )
        return type(self)(**kwargs | overrides)

    def reverse(self) -> Self:
        return self.clone()

    def forward(self, rays: RayBundle, tf: Tf) -> LightTargetRecord:

        # Perform collision detection with the surface
        sout = self.surface(rays.P, rays.V, tf)
        combined = rays.valid & sout.valid

        # check no rays special case after we've performed surface collision
        # so we can still render the surface
        if combined.sum() == 0:
            loss = torch.zeros((), dtype=rays.dtype, device=rays.device)
            return LightTargetRecord(loss, sout)

        # TODO 2D only for now
        if rays.V.shape[-1] == 3:
            loss = torch.zeros((), dtype=rays.dtype, device=rays.device)
            return LightTargetRecord(loss, sout)

        # Compute image surface coordinates here
        # To make this work with any surface, we would need a way to compute
        # surface coordinates for points on a surface, for any surface
        # For a plane it's easy though

        rays_propagated = rays.replace(P=sout.points_global)
        rays_object = rays_propagated.filter(combined).field.values
        rays_image = sout.points_local[combined, 1]

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

        return LightTargetRecord(loss, sout)

    def trace(self, trace: OpticalTrace, key: str, upstream_key: str) -> None:
        upstream = trace.nodes[upstream_key]
        record = self(upstream.bundle_out, upstream.tf_out)
        trace.append(
            key=key,
            record=record,
            module=self,
            upstream={upstream_key},
            bundle_in=upstream.bundle_out,
            tf_in=upstream.tf_out,
            new_bundle=None,
            new_tf=None,
        )
