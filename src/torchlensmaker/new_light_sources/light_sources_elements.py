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

from typing import Any
from jaxtyping import Float

from torchlensmaker.core.tensor_manip import to_tensor
from torchlensmaker.optical_data import OpticalData

from torchlensmaker.elements.sequential import SequentialElement

from torchlensmaker.elements.light_sources import LightSourceBase


# TODO remove forward() of LightSourceBase
class GenericLightSource(LightSourceBase):
    def __init__(
        self,
        pupil_sampler: nn.Module,
        field_sampler: nn.Module,
        wavel_sampler: nn.Module,
        material: nn.Module,
        geometry: nn.Module,
    ):
        super().__init__()
        self.pupil_sampler = pupil_sampler
        self.field_sampler = field_sampler
        self.wavel_sampler = wavel_sampler
        self.material = material
        self.geometry = geometry

    def domain(self) -> dict[str, list[float]]:
        return self.geometry.domain()

    def forward(self, data: OpticalData) -> OpticalData:
        dtype, device = data.dtype, torch.device("cpu")  # TODO gpu support

        # Compute pupil, field and wavelength samples
        pupil_samples = self.pupil_sampler(dtype, device)
        field_samples = self.field_sampler(dtype, device)
        wavel_samples = self.wavel_sampler(dtype, device)

        # Compute rays with the object geometry
        P, V, W, pupil_coords, field_coords = self.geometry(
            pupil_samples,
            field_samples,
            wavel_samples,
        )

        # Compute refraction index with material model
        R = self.material(W)

        return data.replace(
            P=P,
            V=V,
            rays_wavelength=W,
            rays_index=R,
            rays_base=pupil_coords,
            rays_object=field_coords,
        )
