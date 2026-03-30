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
from jaxtyping import Float

from torchlensmaker.core.base_module import BaseModule
from torchlensmaker.core.ray_bundle import RayBundle
from torchlensmaker.light_sources.source_geometry_elements import (
    ObjectAtInfinityGeometry2D,
    ObjectAtInfinityGeometry3D,
    ObjectGeometry2D,
    ObjectGeometry3D,
)
from torchlensmaker.sampling.sampler_elements import (
    DiskSampler2D,
    LinspaceSampler1D,
    ZeroSampler1D,
    ZeroSampler2D,
)
from torchlensmaker.sequential.model_trace import ModelTrace
from torchlensmaker.sequential.sequential_data import SequentialData
from torchlensmaker.types import HomMatrix


class LightSourceBase(BaseModule):
    def domain(self, dim: int) -> dict[str, list[float]]:
        raise NotImplementedError

    def forward(self, tf: HomMatrix) -> RayBundle:
        raise NotImplementedError

    def sequential(self, data: SequentialData) -> tuple[SequentialData, Any, Any]:
        # Merge this light source rays with any previous rays
        new_rays = data.rays.cat(self(data.fk.direct))
        return data.replace(rays=new_rays), data.fk.direct, new_rays

    def trace(self, trace: ModelTrace, key: str, inputs: Any, outputs: Any) -> None:
        trace.add_output_rays(key, outputs)


class GenericLightSource(LightSourceBase):
    def __init__(
        self,
        sampler_pupil_2d: nn.Module,
        sampler_pupil_3d: nn.Module,
        sampler_field_2d: nn.Module,
        sampler_field_3d: nn.Module,
        sampler_wavel_2d: nn.Module,
        sampler_wavel_3d: nn.Module,
        geometry_2d: nn.Module,
        geometry_3d: nn.Module,
        reversed: bool = False,
        source_idx: int = 0,
    ):
        super().__init__()
        self.sampler_pupil_2d = sampler_pupil_2d
        self.sampler_field_2d = sampler_field_2d
        self.sampler_wavel_2d = sampler_wavel_2d
        self.sampler_pupil_3d = sampler_pupil_3d
        self.sampler_field_3d = sampler_field_3d
        self.sampler_wavel_3d = sampler_wavel_3d
        self.geometry_2d = geometry_2d
        self.geometry_3d = geometry_3d
        self.reversed = reversed
        self.source_idx: int = source_idx

    def reverse(self) -> Self:
        return self.clone(reversed=not self.reversed)

    def domain(self, dim: int) -> dict[str, list[float]]:
        if dim == 2:
            return self.geometry_2d.domain()
        else:
            return self.geometry_3d.domain()

    def forward(self, tf: HomMatrix) -> RayBundle:
        dim, dtype, device = tf.shape[0] - 1, tf.dtype, tf.device

        if dim == 2:
            sampler_pupil = self.sampler_pupil_2d
            sampler_field = self.sampler_field_2d
            sampler_wavel = self.sampler_wavel_2d
            geometry = self.geometry_2d
        else:
            sampler_pupil = self.sampler_pupil_3d
            sampler_field = self.sampler_field_3d
            sampler_wavel = self.sampler_wavel_3d
            geometry = self.geometry_3d

        # Compute pupil, field and wavelength samples
        pupil_samples = sampler_pupil(dtype, device)
        field_samples = sampler_field(dtype, device)
        wavel_samples = sampler_wavel(dtype, device)

        # Compute rays with the object geometry
        (
            P,
            V,
            pupil_coords,
            field_coords,
            wavel_coords,
            pupil_idx,
            field_idx,
            wavel_idx,
        ) = geometry(tf, pupil_samples, field_samples, wavel_samples)

        Nrays = P.shape[0]
        assert wavel_coords.shape == (Nrays,), wavel_coords.shape

        # If reversed, flip rays traveling direction
        if self.reversed:
            V = -V

        # Source idx is user provided
        source_idx = torch.full(
            pupil_idx.shape, self.source_idx, dtype=torch.int64, device=device
        )

        return RayBundle.create(
            P=P,
            V=V,
            pupil=pupil_coords,
            field=field_coords,
            wavel=wavel_coords,
            pupil_idx=pupil_idx,
            field_idx=field_idx,
            wavel_idx=wavel_idx,
            source_idx=source_idx,
        )


class Object(GenericLightSource):
    def __init__(
        self,
        beam_angular_size: Float[torch.Tensor, ""] | float,
        object_diameter: Float[torch.Tensor, ""] | float,
        wavelength: int | float | tuple[int | float, int | float] = 500,
        sampler_pupil_2d: nn.Module = LinspaceSampler1D(5),
        sampler_field_2d: nn.Module = LinspaceSampler1D(5),
        sampler_wavel_2d: nn.Module = LinspaceSampler1D(5),
        sampler_pupil_3d: nn.Module = DiskSampler2D(5, 5),
        sampler_field_3d: nn.Module = DiskSampler2D(5, 5),
        sampler_wavel_3d: nn.Module = LinspaceSampler1D(5),
        reversed: bool = False,
        source_idx: int = 0,
    ):
        super().__init__(
            sampler_pupil_2d=sampler_pupil_2d,
            sampler_pupil_3d=sampler_pupil_3d,
            sampler_field_2d=sampler_field_2d,
            sampler_field_3d=sampler_field_3d,
            sampler_wavel_2d=sampler_wavel_2d,
            sampler_wavel_3d=sampler_wavel_3d,
            geometry_2d=ObjectGeometry2D(
                beam_angular_size, object_diameter, wavelength
            ),
            geometry_3d=ObjectGeometry3D(
                beam_angular_size, object_diameter, wavelength
            ),
            reversed=reversed,
            source_idx=source_idx,
        )

    def clone(self, **overrides: Any) -> Self:
        kwargs: dict[str, Any] = dict(
            beam_angular_size=self.geometry_2d.beam_angular_size,
            object_diameter=self.geometry_2d.object_diameter,
            wavelength=torch.stack(
                (self.geometry_2d.wavelength_lower, self.geometry_2d.wavelength_upper)
            ),
            sampler_pupil_2d=self.sampler_pupil_2d,
            sampler_field_2d=self.sampler_field_2d,
            sampler_wavel_2d=self.sampler_wavel_2d,
            sampler_pupil_3d=self.sampler_pupil_3d,
            sampler_field_3d=self.sampler_field_3d,
            sampler_wavel_3d=self.sampler_wavel_3d,
            reversed=self.reversed,
            source_idx=self.source_idx,
        )
        return type(self)(**kwargs | overrides)


class ObjectAtInfinity(GenericLightSource):
    def __init__(
        self,
        beam_diameter: Float[torch.Tensor, ""] | float,
        angular_size: Float[torch.Tensor, ""] | float,
        wavelength: int | float | tuple[int | float, int | float] = 500,
        sampler_pupil_2d: nn.Module = LinspaceSampler1D(5),
        sampler_field_2d: nn.Module = LinspaceSampler1D(5),
        sampler_wavel_2d: nn.Module = LinspaceSampler1D(5),
        sampler_pupil_3d: nn.Module = DiskSampler2D(5, 5),
        sampler_field_3d: nn.Module = DiskSampler2D(5, 5),
        sampler_wavel_3d: nn.Module = LinspaceSampler1D(5),
        reversed: bool = False,
        source_idx: int = 0,
    ):
        super().__init__(
            sampler_pupil_2d=sampler_pupil_2d,
            sampler_field_2d=sampler_field_2d,
            sampler_wavel_2d=sampler_wavel_2d,
            sampler_pupil_3d=sampler_pupil_3d,
            sampler_field_3d=sampler_field_3d,
            sampler_wavel_3d=sampler_wavel_3d,
            geometry_2d=ObjectAtInfinityGeometry2D(
                beam_diameter, angular_size, wavelength
            ),
            geometry_3d=ObjectAtInfinityGeometry3D(
                beam_diameter, angular_size, wavelength
            ),
            reversed=reversed,
            source_idx=source_idx,
        )

    def clone(self, **overrides: Any) -> Self:
        kwargs: dict[str, Any] = dict(
            beam_diameter=self.geometry_2d.beam_diameter,
            angular_size=self.geometry_2d.angular_size,
            wavelength=torch.stack(
                (self.geometry_2d.wavelength_lower, self.geometry_2d.wavelength_upper)
            ),
            sampler_pupil_2d=self.sampler_pupil_2d,
            sampler_field_2d=self.sampler_field_2d,
            sampler_wavel_2d=self.sampler_wavel_2d,
            sampler_pupil_3d=self.sampler_pupil_3d,
            sampler_field_3d=self.sampler_field_3d,
            sampler_wavel_3d=self.sampler_wavel_3d,
            reversed=self.reversed,
            source_idx=self.source_idx,
        )
        return type(self)(**kwargs | overrides)


class PointSource(GenericLightSource):
    def __init__(
        self,
        beam_angular_size: Float[torch.Tensor, ""] | float,
        wavelength: int | float | tuple[int | float, int | float] = 500,
        sampler_pupil_2d: nn.Module = LinspaceSampler1D(5),
        sampler_wavel_2d: nn.Module = LinspaceSampler1D(5),
        sampler_pupil_3d: nn.Module = DiskSampler2D(5, 5),
        sampler_wavel_3d: nn.Module = LinspaceSampler1D(5),
        reversed: bool = False,
        source_idx: int = 0,
    ):
        super().__init__(
            sampler_pupil_2d=sampler_pupil_2d,
            sampler_field_2d=ZeroSampler1D(),
            sampler_wavel_2d=sampler_wavel_2d,
            sampler_pupil_3d=sampler_pupil_3d,
            sampler_field_3d=ZeroSampler2D(),
            sampler_wavel_3d=sampler_wavel_3d,
            geometry_2d=ObjectGeometry2D(
                beam_angular_size=beam_angular_size,
                object_diameter=0,
                wavelength=wavelength,
            ),
            geometry_3d=ObjectGeometry3D(
                beam_angular_size=beam_angular_size,
                object_diameter=0,
                wavelength=wavelength,
            ),
            reversed=reversed,
            source_idx=source_idx,
        )

    def clone(self, **overrides: Any) -> Self:
        kwargs: dict[str, Any] = dict(
            beam_angular_size=self.geometry_2d.beam_angular_size,
            wavelength=torch.stack(
                (self.geometry_2d.wavelength_lower, self.geometry_2d.wavelength_upper)
            ),
            sampler_pupil_2d=self.sampler_pupil_2d,
            sampler_wavel_2d=self.sampler_wavel_2d,
            sampler_pupil_3d=self.sampler_pupil_3d,
            sampler_wavel_3d=self.sampler_wavel_3d,
            reversed=self.reversed,
            source_idx=self.source_idx,
        )
        return type(self)(**kwargs | overrides)


class PointSourceAtInfinity(GenericLightSource):
    def __init__(
        self,
        beam_diameter: Float[torch.Tensor, ""] | float,
        wavelength: int | float | tuple[int | float, int | float] = 500,
        sampler_pupil_2d: nn.Module = LinspaceSampler1D(5),
        sampler_wavel_2d: nn.Module = LinspaceSampler1D(5),
        sampler_pupil_3d: nn.Module = DiskSampler2D(5, 5),
        sampler_wavel_3d: nn.Module = LinspaceSampler1D(5),
        reversed: bool = False,
        source_idx: int = 0,
    ):
        super().__init__(
            sampler_pupil_2d=sampler_pupil_2d,
            sampler_field_2d=ZeroSampler1D(),
            sampler_wavel_2d=sampler_wavel_2d,
            sampler_pupil_3d=sampler_pupil_3d,
            sampler_field_3d=ZeroSampler2D(),
            sampler_wavel_3d=sampler_wavel_3d,
            geometry_2d=ObjectAtInfinityGeometry2D(
                beam_diameter=beam_diameter,
                angular_size=0,
                wavelength=wavelength,
            ),
            geometry_3d=ObjectAtInfinityGeometry3D(
                beam_diameter=beam_diameter,
                angular_size=0,
                wavelength=wavelength,
            ),
            reversed=reversed,
            source_idx=source_idx,
        )

    def clone(self, **overrides: Any) -> Self:
        kwargs: dict[str, Any] = dict(
            beam_diameter=self.geometry_2d.beam_diameter,
            wavelength=torch.stack(
                (self.geometry_2d.wavelength_lower, self.geometry_2d.wavelength_upper)
            ),
            sampler_pupil_2d=self.sampler_pupil_2d,
            sampler_wavel_2d=self.sampler_wavel_2d,
            sampler_pupil_3d=self.sampler_pupil_3d,
            sampler_wavel_3d=self.sampler_wavel_3d,
            reversed=self.reversed,
            source_idx=self.source_idx,
        )
        return type(self)(**kwargs | overrides)


class RaySource(GenericLightSource):
    def __init__(
        self,
        wavelength: int | float | tuple[int | float, int | float] = 500,
        sampler_wavel_2d: nn.Module = LinspaceSampler1D(5),
        sampler_wavel_3d: nn.Module = LinspaceSampler1D(5),
        reversed: bool = False,
        source_idx: int = 0,
    ):
        super().__init__(
            sampler_pupil_2d=ZeroSampler1D(),
            sampler_field_2d=ZeroSampler1D(),
            sampler_wavel_2d=sampler_wavel_2d,
            sampler_pupil_3d=ZeroSampler2D(),
            sampler_field_3d=ZeroSampler2D(),
            sampler_wavel_3d=sampler_wavel_3d,
            geometry_2d=ObjectAtInfinityGeometry2D(
                beam_diameter=0,
                angular_size=0,
                wavelength=wavelength,
            ),
            geometry_3d=ObjectAtInfinityGeometry3D(
                beam_diameter=0,
                angular_size=0,
                wavelength=wavelength,
            ),
            reversed=reversed,
            source_idx=source_idx,
        )

    def clone(self, **overrides: Any) -> Self:
        kwargs: dict[str, Any] = dict(
            wavelength=torch.stack(
                (self.geometry_2d.wavelength_lower, self.geometry_2d.wavelength_upper)
            ),
            sampler_wavel_2d=self.sampler_wavel_2d,
            sampler_wavel_3d=self.sampler_wavel_3d,
            reversed=self.reversed,
            source_idx=self.source_idx,
        )
        return type(self)(**kwargs | overrides)
