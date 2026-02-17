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

from jaxtyping import Float

from torchlensmaker.optical_data import OpticalData

from torchlensmaker.core.dim import Dim
from torchlensmaker.elements.sequential_element import SequentialElement

from torchlensmaker.sampling.sampler_elements import (
    LinspaceSampler1D,
    ZeroSampler1D,
    ZeroSampler2D,
    DiskSampler2D,
)
from torchlensmaker.light_sources.source_geometry_elements import (
    ObjectGeometry2D,
    ObjectAtInfinityGeometry2D,
    ObjectGeometry3D,
    ObjectAtInfinityGeometry3D,
)
from torchlensmaker.materials.material_elements import MaterialModel
from torchlensmaker.materials.get_material_model import get_material_model

from torchlensmaker.kinematics.homogeneous_geometry import transform_rays


# TODO make forward() not sequential
class LightSourceBase(SequentialElement):
    def domain(self, dim: int) -> dict[str, list[float]]:
        raise NotImplementedError

    def dim(self) -> Dim:
        raise NotImplementedError

    def forward(self, data: OpticalData) -> OpticalData:
        raise NotImplementedError


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

    def domain(self, dim: int) -> dict[str, list[float]]:
        return self.geometry.domain()

    def forward(self, data: OpticalData) -> OpticalData:
        dtype, device = data.dtype, torch.device("cpu")  # TODO gpu support

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

        Nrays = P.shape[0]
        assert W.shape == (Nrays,), W.shape

        # Compute refraction index with material model
        R = self.material(W)

        # Apply kinematic transform
        P, V = transform_rays(data.fk.direct, P, V)

        return data.replace(
            P=P,
            V=V,
            rays_wavelength=W,
            rays_index=R,
            rays_pupil=pupil_coords,
            rays_field=field_coords,
        )


class Object2D(GenericLightSource):
    def __init__(
        self,
        beam_angular_size: Float[torch.Tensor, ""] | float,
        object_diameter: Float[torch.Tensor, ""] | float,
        material: str | MaterialModel = "air",
        wavelength: int | float | tuple[int | float, int | float] = 500,
        sampler_pupil: nn.Module = LinspaceSampler1D(5),
        sampler_field: nn.Module = LinspaceSampler1D(5),
        sampler_wavelength: nn.Module = LinspaceSampler1D(5),
    ):
        super().__init__(
            sampler_pupil=sampler_pupil,
            sampler_field=sampler_field,
            sampler_wavelength=sampler_wavelength,
            material=get_material_model(material),
            geometry=ObjectGeometry2D(beam_angular_size, object_diameter, wavelength),
        )

    def dim(self) -> Dim:
        return Dim.TWO


class Object3D(GenericLightSource):
    def __init__(
        self,
        beam_angular_size: Float[torch.Tensor, ""] | float,
        object_diameter: Float[torch.Tensor, ""] | float,
        material: str | MaterialModel = "air",
        wavelength: int | float | tuple[int | float, int | float] = 500,
        sampler_pupil: nn.Module = DiskSampler2D(5, 5),
        sampler_field: nn.Module = DiskSampler2D(5, 5),
        sampler_wavelength: nn.Module = LinspaceSampler1D(5),
    ):
        super().__init__(
            sampler_pupil=sampler_pupil,
            sampler_field=sampler_field,
            sampler_wavelength=sampler_wavelength,
            material=get_material_model(material),
            geometry=ObjectGeometry3D(beam_angular_size, object_diameter, wavelength),
        )

    def dim(self) -> Dim:
        return Dim.THREE


class ObjectAtInfinity2D(GenericLightSource):
    def __init__(
        self,
        beam_diameter: Float[torch.Tensor, ""] | float,
        angular_size: Float[torch.Tensor, ""] | float,
        material: str | MaterialModel = "air",
        wavelength: int | float | tuple[int | float, int | float] = 500,
        sampler_pupil: nn.Module = LinspaceSampler1D(5),
        sampler_field: nn.Module = LinspaceSampler1D(5),
        sampler_wavelength: nn.Module = LinspaceSampler1D(5),
    ):
        super().__init__(
            sampler_pupil=sampler_pupil,
            sampler_field=sampler_field,
            sampler_wavelength=sampler_wavelength,
            material=get_material_model(material),
            geometry=ObjectAtInfinityGeometry2D(
                beam_diameter, angular_size, wavelength
            ),
        )

    def dim(self) -> Dim:
        return Dim.TWO


class ObjectAtInfinity3D(GenericLightSource):
    def __init__(
        self,
        beam_diameter: Float[torch.Tensor, ""] | float,
        angular_size: Float[torch.Tensor, ""] | float,
        material: str | MaterialModel = "air",
        wavelength: int | float | tuple[int | float, int | float] = 500,
        sampler_pupil: nn.Module = DiskSampler2D(5, 5),
        sampler_field: nn.Module = DiskSampler2D(5, 5),
        sampler_wavelength: nn.Module = LinspaceSampler1D(5),
    ):
        super().__init__(
            sampler_pupil=sampler_pupil,
            sampler_field=sampler_field,
            sampler_wavelength=sampler_wavelength,
            material=get_material_model(material),
            geometry=ObjectAtInfinityGeometry3D(
                beam_diameter, angular_size, wavelength
            ),
        )

    def dim(self) -> Dim:
        return Dim.THREE


class PointSource2D(GenericLightSource):
    def __init__(
        self,
        beam_angular_size: Float[torch.Tensor, ""] | float,
        material: str | MaterialModel = "air",
        wavelength: int | float | tuple[int | float, int | float] = 500,
        sampler_pupil: nn.Module = LinspaceSampler1D(5),
        sampler_wavelength: nn.Module = LinspaceSampler1D(5),
    ):
        super().__init__(
            sampler_pupil=sampler_pupil,
            sampler_field=ZeroSampler1D(),
            sampler_wavelength=sampler_wavelength,
            material=get_material_model(material),
            geometry=ObjectGeometry2D(
                beam_angular_size=beam_angular_size,
                object_diameter=0,
                wavelength=wavelength,
            ),
        )

    def dim(self) -> Dim:
        return Dim.TWO


class PointSourceAtInfinity2D(GenericLightSource):
    def __init__(
        self,
        beam_diameter: Float[torch.Tensor, ""] | float,
        material: str | MaterialModel = "air",
        wavelength: int | float | tuple[int | float, int | float] = 500,
        sampler_pupil: nn.Module = LinspaceSampler1D(5),
        sampler_wavelength: nn.Module = LinspaceSampler1D(5),
    ):
        super().__init__(
            sampler_pupil=sampler_pupil,
            sampler_field=ZeroSampler1D(),
            sampler_wavelength=sampler_wavelength,
            material=get_material_model(material),
            geometry=ObjectAtInfinityGeometry2D(
                beam_diameter=beam_diameter,
                angular_size=0,
                wavelength=wavelength,
            ),
        )

    def dim(self) -> Dim:
        return Dim.TWO


class PointSource3D(GenericLightSource):
    def __init__(
        self,
        beam_angular_size: Float[torch.Tensor, ""] | float,
        material: str | MaterialModel = "air",
        wavelength: int | float | tuple[int | float, int | float] = 500,
        sampler_pupil: nn.Module = DiskSampler2D(5, 5),
        sampler_wavelength: nn.Module = LinspaceSampler1D(5),
    ):
        super().__init__(
            sampler_pupil=sampler_pupil,
            sampler_field=ZeroSampler2D(),
            sampler_wavelength=sampler_wavelength,
            material=get_material_model(material),
            geometry=ObjectGeometry3D(
                beam_angular_size=beam_angular_size,
                object_diameter=0,
                wavelength=wavelength,
            ),
        )

    def dim(self) -> Dim:
        return Dim.THREE


class PointSourceAtInfinity3D(GenericLightSource):
    def __init__(
        self,
        beam_diameter: Float[torch.Tensor, ""] | float,
        material: str | MaterialModel = "air",
        wavelength: int | float | tuple[int | float, int | float] = 500,
        sampler_pupil: nn.Module = DiskSampler2D(5, 5),
        sampler_wavelength: nn.Module = LinspaceSampler1D(5),
    ):
        super().__init__(
            sampler_pupil=sampler_pupil,
            sampler_field=ZeroSampler2D(),
            sampler_wavelength=sampler_wavelength,
            material=get_material_model(material),
            geometry=ObjectAtInfinityGeometry3D(
                beam_diameter=beam_diameter,
                angular_size=0,
                wavelength=wavelength,
            ),
        )

    def dim(self) -> Dim:
        return Dim.THREE


class RaySource2D(GenericLightSource):
    def __init__(
        self,
        material: str | MaterialModel = "air",
        wavelength: int | float | tuple[int | float, int | float] = 500,
        sampler_wavelength: nn.Module = LinspaceSampler1D(5),
    ):
        super().__init__(
            sampler_pupil=ZeroSampler1D(),
            sampler_field=ZeroSampler1D(),
            sampler_wavelength=sampler_wavelength,
            material=get_material_model(material),
            geometry=ObjectAtInfinityGeometry2D(
                beam_diameter=0,
                angular_size=0,
                wavelength=wavelength,
            ),
        )

    def dim(self) -> Dim:
        return Dim.TWO


class RaySource3D(GenericLightSource):
    def __init__(
        self,
        material: str | MaterialModel = "air",
        wavelength: int | float | tuple[int | float, int | float] = 500,
        sampler_wavelength: nn.Module = LinspaceSampler1D(5),
    ):
        super().__init__(
            sampler_pupil=ZeroSampler2D(),
            sampler_field=ZeroSampler2D(),
            sampler_wavelength=sampler_wavelength,
            material=get_material_model(material),
            geometry=ObjectAtInfinityGeometry3D(
                beam_diameter=0,
                angular_size=0,
                wavelength=wavelength,
            ),
        )

    def dim(self) -> Dim:
        return Dim.THREE


class MixedDimLightSource(LightSourceBase):
    def __init__(self, module_2d: nn.Module, module_3d: nn.Module):
        super().__init__()
        self.module_2d = module_2d
        self.module_3d = module_3d

    def domain(self, dim: int) -> dict[str, list[float]]:
        if dim == 2:
            return self.module_2d.domain(dim)
        else:
            return self.module_3d.domain(dim)

    def dim(self) -> Dim:
        return Dim.MIXED

    def forward(self, data: OpticalData) -> OpticalData:
        if data.dim == 2:
            return self.module_2d(data)
        else:
            return self.module_3d(data)


class Object(MixedDimLightSource):
    def __init__(self, *args, **kwargs):
        super().__init__(Object2D(*args, **kwargs), Object3D(*args, **kwargs))


class ObjectAtInfinity(MixedDimLightSource):
    def __init__(self, *args, **kwargs):
        super().__init__(
            ObjectAtInfinity2D(*args, **kwargs), ObjectAtInfinity3D(*args, **kwargs)
        )


class PointSource(MixedDimLightSource):
    def __init__(self, *args, **kwargs):
        super().__init__(PointSource2D(*args, **kwargs), PointSource3D(*args, **kwargs))


class PointSourceAtInfinity(MixedDimLightSource):
    def __init__(self, *args, **kwargs):
        super().__init__(
            PointSourceAtInfinity2D(*args, **kwargs),
            PointSourceAtInfinity3D(*args, **kwargs),
        )


class RaySource(MixedDimLightSource):
    def __init__(self, *args, **kwargs):
        super().__init__(RaySource2D(*args, **kwargs), RaySource3D(*args, **kwargs))
