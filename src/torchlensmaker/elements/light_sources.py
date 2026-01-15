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

from torchlensmaker.core.tensor_manip import (
    cat_optional,
    cartesian_prod2d,
    to_tensor,
)
from torchlensmaker.kinematics.homogeneous_geometry import transform_rays
from torchlensmaker.optical_data import OpticalData
from torchlensmaker.elements.sequential import SequentialElement

from torchlensmaker.core.geometry import unit_vector, rotated_unit_vector

from torchlensmaker.sampling.samplers import (
    sampleND,
)

from torchlensmaker.materials import MaterialModel, get_material_model


from typing import Any, Optional

Tensor = torch.Tensor


class LightSourceBase(SequentialElement):
    def __init__(self, material: str | MaterialModel = "air"):
        super().__init__()
        self.material = get_material_model(material)

    def sample_light_source(
        self, sampling: dict[str, Any], dim: int, dtype: torch.dtype
    ) -> tuple[
        Tensor,
        Tensor,
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
    ]:
        raise NotImplementedError

    def forward(self, inputs: OpticalData) -> OpticalData:
        dim, dtype = inputs.dim, inputs.dtype

        # Get samples from derived class in local frame
        P, V, rays_base, rays_object, var_base, var_object = self.sample_light_source(
            inputs.sampling, dim, dtype
        )

        # Apply kinematic transform
        P, V = transform_rays(inputs.dfk, P, V)

        # TODO need better way to store var basis if we want to properly support multiple light source in a scene

        return inputs.replace(
            P=torch.cat((inputs.P, P), dim=0),
            V=torch.cat((inputs.V, V), dim=0),
            rays_base=cat_optional(inputs.rays_base, rays_base),
            rays_object=cat_optional(inputs.rays_object, rays_object),
            var_base=var_base,
            var_object=var_object,
            material=self.material,
        )


class RaySource(LightSourceBase):
    def sample_light_source(
        self, sampling: dict[str, Any], dim: int, dtype: torch.dtype
    ) -> tuple[
        Tensor,
        Tensor,
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
    ]:
        P = torch.zeros(1, dim, dtype=dtype)
        V = unit_vector(dim, dtype).unsqueeze(0)

        return P, V, None, None, None, None


class PointSourceAtInfinity(LightSourceBase):
    def __init__(self, beam_diameter: float, **kwargs: Any):
        super().__init__(**kwargs)
        self.beam_diameter: Tensor = to_tensor(beam_diameter)

    def sample_light_source(
        self, sampling: dict[str, Any], dim: int, dtype: torch.dtype
    ) -> tuple[
        Tensor,
        Tensor,
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
    ]:
        # Sample coordinates other than X on a disk
        NX = sampleND(
            sampling["base"],
            self.beam_diameter,
            dim - 1,
            dtype,
        )

        # Make the rays P + tV
        P = torch.column_stack((torch.zeros(NX.shape[0], dtype=dtype), NX))
        V = unit_vector(dim, dtype).unsqueeze(0)

        # Cartesian product
        fullP, fullV = cartesian_prod2d(P, V)

        return fullP, fullV, NX, None, NX, None


class PointSource(LightSourceBase):
    def __init__(self, beam_angular_size: float, **kwargs: Any):
        super().__init__(**kwargs)

        self.beam_angular_size = torch.deg2rad(to_tensor(beam_angular_size))

    def sample_light_source(
        self, sampling: dict[str, Any], dim: int, dtype: torch.dtype
    ) -> tuple[
        Tensor,
        Tensor,
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
    ]:
        # Sample angular direction
        angles = sampleND(
            sampling["base"],
            self.beam_angular_size,
            dim - 1,
            dtype,
        )

        V = rotated_unit_vector(angles, dim)
        P = torch.zeros((1, dim), dtype=dtype)

        # Cartesian product
        fullP, fullV = cartesian_prod2d(P, V)

        return fullP, fullV, angles, None, angles, None


class ObjectAtInfinity(LightSourceBase):
    def __init__(self, beam_diameter: float, angular_size: float, **kwargs: Any):
        super().__init__(**kwargs)
        self.beam_diameter: Tensor = to_tensor(beam_diameter)
        self.angular_size: Tensor = torch.deg2rad(to_tensor(angular_size))

    def sample_light_source(
        self, sampling: dict[str, Any], dim: int, dtype: torch.dtype
    ) -> tuple[
        Tensor,
        Tensor,
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
    ]:
        # Sample coordinates other than X on a disk
        NX = sampleND(
            sampling["base"],
            self.beam_diameter,
            dim - 1,
            dtype,
        )
        P = torch.column_stack((torch.zeros(NX.shape[0], dtype=dtype), NX))

        # Sample angular direction
        angles = sampleND(
            sampling["object"],
            self.angular_size,
            dim - 1,
            dtype,
        )
        V = rotated_unit_vector(angles, dim)

        # Cartesian product
        fullP, fullV = cartesian_prod2d(P, V)
        rays_base, rays_object = cartesian_prod2d(NX, angles)

        return fullP, fullV, rays_base, rays_object, NX, angles


class Object(LightSourceBase):
    def __init__(self, beam_angular_size: float, object_diameter: float, **kwargs: Any):
        super().__init__(**kwargs)
        self.beam_angular_size: Tensor = torch.deg2rad(to_tensor(beam_angular_size))
        self.object_diameter: Tensor = to_tensor(object_diameter)

    def sample_light_source(
        self, sampling: dict[str, Any], dim: int, dtype: torch.dtype
    ) -> tuple[
        Tensor,
        Tensor,
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
    ]:
        # Sample coordinates other than X on a disk
        NX = sampleND(
            sampling["object"],
            self.object_diameter,
            dim - 1,
            dtype,
        )
        P = torch.column_stack((torch.zeros(NX.shape[0], dtype=dtype), NX))

        # Sample angular direction
        angles = sampleND(
            sampling["base"],
            self.beam_angular_size,
            dim - 1,
            dtype,
        )
        V = rotated_unit_vector(angles, dim)

        # Cartesian product
        fullP, fullV = cartesian_prod2d(P, V)
        rays_object, rays_base = cartesian_prod2d(NX, angles)

        return fullP, fullV, rays_base, rays_object, angles, NX


def cartesian_wavelength(inputs: OpticalData, ray_var: Tensor) -> OpticalData:
    "Add wavelength var by doing cartesian product with existing vars"

    N = inputs.P.shape[0]
    M = ray_var.shape[0]

    new_P = torch.repeat_interleave(inputs.P, M, dim=0)
    new_V = torch.repeat_interleave(inputs.V, M, dim=0)
    new_rays_base = (
        torch.repeat_interleave(inputs.rays_base, M, dim=0)
        if inputs.rays_base is not None
        else None
    )
    new_rays_object = (
        torch.repeat_interleave(inputs.rays_object, M, dim=0)
        if inputs.rays_object is not None
        else None
    )

    new_var = torch.tile(ray_var, (N,))

    return inputs.replace(
        P=new_P,
        V=new_V,
        rays_base=new_rays_base,
        rays_object=new_rays_object,
        rays_wavelength=new_var,
    )


class Wavelength(SequentialElement):
    def __init__(self, lower: float | int, upper: float | int):
        super().__init__()
        self.lower, self.upper = to_tensor(lower), to_tensor(upper)

    def forward(self, inputs: OpticalData) -> OpticalData:
        if inputs.rays_wavelength is not None:
            raise RuntimeError(
                "Rays already have wavelength data. Cannot apply Wavelength()."
            )

        if "wavelength" not in inputs.sampling:
            raise RuntimeError(
                "Missing 'wavelength' key in sampling configuration. Cannot apply Wavelength()."
            )

        chromatic_space = inputs.sampling["wavelength"].sample1d(
            self.lower, self.upper, inputs.dtype
        )

        return cartesian_wavelength(inputs, chromatic_space).replace(
            var_wavelength=chromatic_space
        )
