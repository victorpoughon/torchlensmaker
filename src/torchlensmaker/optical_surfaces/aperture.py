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


from typing import Any, Self

from torchlensmaker.core.ray_bundle import RayBundle
from torchlensmaker.elements.sequential_data import SequentialData
from torchlensmaker.surfaces.surface_disk import Disk
from torchlensmaker.types import Direction, ScalarTensor, Tf

from .optical_surface import OpticalSurfaceElement
from .surface_propagator import SurfacePropagator


class Aperture(OpticalSurfaceElement):
    def __init__(self, diameter: float | ScalarTensor):
        super().__init__()
        self.propagator = SurfacePropagator(Disk(diameter))

    def clone(self, **overrides: Any) -> Self:
        kwargs: dict[str, Any] = dict(diameter=self.propagator.surface.diameter)
        return type(self)(**kwargs | overrides)

    def sequential(self, inputs: SequentialData) -> SequentialData:
        rays_propagated, tf_surface, fk_next = self(
            inputs.rays, inputs.fk, inputs.direction
        )
        return inputs.replace(
            rays=rays_propagated,
            fk=fk_next,  # correct but useless cause Aperture is only ever a disk currently
        )

    def forward(
        self, rays: RayBundle, tf: Tf, direction: Direction
    ) -> tuple[RayBundle, Tf, Tf]:
        rays_propagated, _, tf_surface, fk_next = self.propagator(rays, tf, direction)

        return rays_propagated, tf_surface, fk_next
