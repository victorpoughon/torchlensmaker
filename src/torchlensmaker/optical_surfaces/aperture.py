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
from torchlensmaker.surfaces import Disk, SurfaceElementOutput
from torchlensmaker.types import ScalarTensor, Tf

from .optical_surface import OpticalSurfaceElement


class Aperture(OpticalSurfaceElement):
    def __init__(self, diameter: float | ScalarTensor):
        super().__init__()
        self.surface = Disk(diameter)

    def clone(self, **overrides: Any) -> Self:
        kwargs: dict[str, Any] = dict(diameter=self.surface.diameter)
        return type(self)(**kwargs | overrides)

    def __repr__(self) -> str:
        return f"{self._get_name()}(diameter={self.surface.diameter.item()})"

    def reverse(self) -> Self:
        return self.clone()

    def forward(
        self, rays: RayBundle, tf: Tf
    ) -> tuple[RayBundle, SurfaceElementOutput]:
        sout = self.surface(rays.P, rays.V, tf)
        # TODO should probably propagate for consistency?
        new_rays = rays.mask(sout.valid)
        return new_rays, sout
