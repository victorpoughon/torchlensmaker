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
import torch.nn as nn

from torchlensmaker.core.ray_bundle import RayBundle
from torchlensmaker.materials.get_material_model import (
    get_material_model,
)
from torchlensmaker.materials.material_elements import MaterialModel
from torchlensmaker.physics.physics_kernels import RefractionKernel
from torchlensmaker.surfaces.surface_element import SurfaceElement, SurfaceElementOutput
from torchlensmaker.types import Tf, TIRMode

from .optical_surface import OpticalSurfaceElement


class RefractiveSurface(OpticalSurfaceElement):
    def __init__(
        self,
        surface: SurfaceElement,
        materials: tuple[str | MaterialModel, str | MaterialModel],
        tir_mode: TIRMode = "absorb",
    ):
        super().__init__()
        self.surface = surface
        self.func = RefractionKernel()
        self.material_in = get_material_model(materials[0]).clone()
        self.material_out = get_material_model(materials[1]).clone()
        self.tir_mode: TIRMode = tir_mode

    @property
    def materials(self) -> tuple[MaterialModel, MaterialModel]:
        return (self.material_in, self.material_out)

    def clone(self, **overrides: Any) -> Self:
        kwargs: dict[str, Any] = dict(
            surface=self.surface,
            materials=self.materials,
            tir_mode=self.tir_mode,
        )
        return type(self)(**kwargs | overrides)

    def reverse(self) -> Self:
        return self.clone(
            surface=self.surface.reverse(),
            materials=(self.material_out, self.material_in),
        )

    def forward(
        self, rays: RayBundle, tf: Tf
    ) -> tuple[RayBundle, SurfaceElementOutput]:
        # Raytrace with the surface
        sout = self.surface(rays.P, rays.V, tf)

        # Compute indices of refraction
        n1 = self.material_in(rays.wavel)
        n2 = self.material_out(rays.wavel)
        assert n1.shape == n2.shape == (rays.batch_size)
        assert n1.device == n2.device

        # Compute optical refraction -- Snell's law happens here
        # Note that some rays are invalid collisions here, we compute refraction anyway
        # as they will be filtered by the combined mask below
        refracted, valid_refraction = self.func.apply(rays.V, sout.normals, n1, n2)

        # The combined valid mask
        if self.tir_mode == "reflect":
            valid = sout.valid
        else:
            valid = torch.logical_and(sout.valid, valid_refraction)

        # Filter the ray bundle for valid rays and apply new vector computed by refraction
        rays_refracted = rays.mask(valid).replace(
            P=sout.points_global[valid], V=refracted[valid]
        )

        return rays_refracted, sout
