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

from torchlensmaker.new_sampling.sampler_elements import (
    LinspaceSampler1D,
    LinspaceSampler2D,
    ZeroSampler1D,
    ZeroSampler2D,
    ExactSampler1D,
)
from torchlensmaker.new_light_sources.source_geometry_elements import (
    ObjectGeometry2D,
    ObjectAtInfinityGeometry2D,
)
from torchlensmaker.new_material.material_elements import MaterialModel
from torchlensmaker.new_material.get_material_model import get_material_model

from torchlensmaker.kinematics.homogeneous_geometry import transform_rays


def convert_sampler_old_to_new(current: nn.Module, value: Any, dtype: torch.dtype, device: torch.device):
    if isinstance(current, ZeroSampler1D):
        return current
    
    if isinstance(value, (float, int)):
        return LinspaceSampler1D(value)
    elif isinstance(value, (list, tuple)):
        return ExactSampler1D(torch.tensor(value, dtype=dtype, device=device))
    else:
        raise RuntimeError(
            f"Sampling: expected number or list of numbers, got {type(value)}: {value}"
        )


# TODO remove LightSourceBase
class GenericLightSource(LightSourceBase):
    def __init__(
        self,
        sampler_pupil: nn.Module,
        sampler_field: nn.Module,
        sampler_wavelength: nn.Module,
        material: nn.Module,
        geometry: nn.Module,
    ):
        super().__init__()
        self.sampler_pupil = sampler_pupil
        self.sampler_field = sampler_field
        self.sampler_wavelength = sampler_wavelength
        self.material = material
        self.geometry = geometry

    def domain(self) -> dict[str, list[float]]:
        return self.geometry.domain()

    def forward(self, data: OpticalData) -> OpticalData:
        dtype, device = data.dtype, torch.device("cpu")  # TODO gpu support

        # TODO improve sampling dict TLM-80
        # for now set the parameters here
        self.sampler_pupil = convert_sampler_old_to_new(
            self.sampler_pupil,
            data.sampling["base"], dtype, device
        )
        self.sampler_field = convert_sampler_old_to_new(
            self.sampler_field,
            data.sampling["object"], dtype, device
        )
        self.sampler_wavelength = convert_sampler_old_to_new(
            self.sampler_wavelength,
            data.sampling["wavelength"], dtype, device
        )

        # Compute pupil, field and wavelength samples
        pupil_samples = self.sampler_pupil(dtype, device)
        field_samples = self.sampler_field(dtype, device)
        wavel_samples = self.sampler_wavelength(dtype, device)

        # Compute rays with the object geometry
        P, V, W, pupil_coords, field_coords = self.geometry(
            pupil_samples,
            field_samples,
            wavel_samples,
        )

        # Compute refraction index with material model
        R = self.material(W)

        # Apply kinematic transform
        P, V = transform_rays(data.dfk, P, V)

        return data.replace(
            P=P,
            V=V,
            rays_wavelength=W,
            rays_index=R,
            rays_base=pupil_coords,
            rays_object=field_coords,
        )


class Object2D(GenericLightSource):
    def __init__(
        self,
        beam_angular_size: Float[torch.Tensor, ""] | float | int,
        object_diameter: Float[torch.Tensor, ""] | float | int,
        material: str | MaterialModel = "air",
        wavelength: int | float | tuple[int | float, int | float] = 500,
    ):
        super().__init__(
            sampler_pupil=LinspaceSampler1D(5),
            sampler_field=LinspaceSampler1D(5),
            sampler_wavelength=LinspaceSampler1D(5),
            material=get_material_model(material),
            geometry=ObjectGeometry2D(beam_angular_size, object_diameter, wavelength),
        )
        # TODO how to setup samplers params?


class ObjectAtInfinity2D(GenericLightSource):
    def __init__(
        self,
        beam_diameter: Float[torch.Tensor, ""] | float | int,
        angular_size: Float[torch.Tensor, ""] | float | int,
        material: str | MaterialModel = "air",
        wavelength: int | float | tuple[int | float, int | float] = 500,
    ):
        super().__init__(
            sampler_pupil=LinspaceSampler1D(5),
            sampler_field=LinspaceSampler1D(5),
            sampler_wavelength=LinspaceSampler1D(5),
            material=get_material_model(material),
            geometry=ObjectAtInfinityGeometry2D(
                beam_diameter, angular_size, wavelength
            ),
        )
        # TODO how to setup samplers params?


class PointSource2D(GenericLightSource):
    def __init__(
        self,
        beam_angular_size: Float[torch.Tensor, ""] | float | int,
        material: str | MaterialModel = "air",
        wavelength: int | float | tuple[int | float, int | float] = 500,
    ):
        super().__init__(
            sampler_pupil=LinspaceSampler1D(5),
            sampler_field=ZeroSampler1D(),
            sampler_wavelength=LinspaceSampler1D(5),
            material=get_material_model(material),
            geometry=ObjectGeometry2D(
                beam_angular_size=beam_angular_size,
                object_diameter=0,
                wavelength=wavelength,
            ),
        )
        # TODO how to setup samplers params?


class PointSourceAtInfinity2D(GenericLightSource):
    def __init__(
        self,
        beam_diameter: Float[torch.Tensor, ""] | float | int,
        material: str | MaterialModel = "air",
        wavelength: int | float | tuple[int | float, int | float] = 500,
    ):
        super().__init__(
            sampler_pupil=LinspaceSampler1D(5),
            sampler_field=ZeroSampler1D(),
            sampler_wavelength=LinspaceSampler1D(5),
            material=get_material_model(material),
            geometry=ObjectAtInfinityGeometry2D(
                beam_diameter=beam_diameter,
                angular_size=0,
                wavelength=wavelength,
            ),
        )
        # TODO how to setup samplers params?


class RaySource2D(GenericLightSource):
    def __init__(
        self,
        material: str | MaterialModel = "air",
        wavelength: int | float | tuple[int | float, int | float] = 500,
    ):
        super().__init__(
            sampler_pupil=ZeroSampler1D(),
            sampler_field=ZeroSampler1D(),
            sampler_wavelength=LinspaceSampler1D(5),
            material=get_material_model(material),
            geometry=ObjectAtInfinityGeometry2D(
                beam_diameter=0,
                angular_size=0,
                wavelength=wavelength,
            ),
        )
        # TODO how to setup samplers params?
