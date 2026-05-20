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

import torch

from torchlensmaker.core.ray_bundle import RayBundle
from torchlensmaker.physics.physics_kernels import ReflectionKernel
from torchlensmaker.surfaces import SurfaceElement
from torchlensmaker.types import Tf

from .optical_surface import OpticalSurfaceElement, OpticalSurfaceRecord


class ReflectiveSurface(OpticalSurfaceElement):
    def __init__(self, surface: SurfaceElement):
        super().__init__()
        self.surface = surface
        self.func = ReflectionKernel()

    def clone(self, **overrides: Any) -> Self:
        kwargs: dict[str, Any] = dict(surface=self.surface)
        return type(self)(**kwargs | overrides)

    def reverse(self) -> Self:
        return self.clone(surface=self.surface.reverse())

    def forward(self, rays: RayBundle, tf: Tf) -> OpticalSurfaceRecord:
        # Raytrace with the surface
        sout = self.surface(rays.P, rays.V, tf)

        # Compute optical reflection
        reflected = self.func.apply(rays.V, sout.normals)

        # avoid nan values propagating to preserve gradients
        combined = rays.valid & sout.valid
        reflected = torch.where(combined.unsqueeze(-1), reflected, rays.V)

        # Filter the ray bundle for valid collisions
        points = sout.points_global
        rays_reflected = rays.mask(sout.valid).replace(P=points, V=reflected)

        return OpticalSurfaceRecord(rays_reflected, sout)
