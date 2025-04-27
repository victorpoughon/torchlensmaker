# This file is part of Torch Lens Maker
# Copyright (C) 2025 Victor Poughon
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

from dataclasses import dataclass, replace

from typing import Any, Optional, TypeAlias

from torchlensmaker.core.tensor_manip import filter_optional_tensor

from torchlensmaker.core.transforms import (
    TransformBase,
    IdentityTransform,
    forward_kinematic,
)

from torchlensmaker.materials import (
    MaterialModel,
    get_material_model,
)

from torchlensmaker.sampling.samplers import Sampler, init_sampling

Tensor: TypeAlias = torch.Tensor


@dataclass
class OpticalData:
    # dim is 2 or 3
    # dtype default is torch.float64
    dim: int
    dtype: torch.dtype

    # Sampling configuration for each variable
    sampling: dict[str, Sampler]

    # Transform kinematic chain
    transforms: list[TransformBase]

    # Parametric light rays P + tV
    # Tensors of shape (N, 2|3)
    P: Tensor
    V: Tensor

    # Rays variables
    # Tensors of shape (N, 2|3) or None
    rays_base: Optional[Tensor]
    rays_object: Optional[Tensor]
    rays_image: Optional[Tensor]
    rays_wavelength: Optional[Tensor]

    # Basis of each sampling variable
    # Tensors of shape (*, 2|3)
    # number of rows is the size of each sampling dimension
    var_base: Optional[Tensor]
    var_object: Optional[Tensor]
    var_wavelength: Optional[Tensor]

    # Material model for this batch of rays
    material: MaterialModel

    # Loss accumulator
    # Tensor of dim 0
    loss: torch.Tensor

    def tf(self) -> TransformBase:
        return forward_kinematic(self.transforms)

    def target(self) -> Tensor:
        return self.tf().direct_points(torch.zeros((self.dim,), dtype=self.dtype))

    def replace(self, /, **changes: Any) -> "OpticalData":
        return replace(self, **changes)

    def get_var_optional(self, name: str) -> Optional[Tensor]:
        if name == "base":
            return self.var_base
        elif name == "object":
            return self.var_object
        elif name == "wavelength":
            return self.var_wavelength
        else:
            raise RuntimeError("Unknown ray variable '{color_dim}'")

    def get_rays(self, color_dim: str) -> Tensor:
        if color_dim == "base" and self.rays_base is not None:
            return self.rays_base
        elif color_dim == "object" and self.rays_object is not None:
            return self.rays_object
        elif color_dim == "wavelength" and self.rays_wavelength is not None:
            return self.rays_wavelength
        else:
            raise RuntimeError(f"Unknown or unavailable ray variable '{color_dim}'")
    
    def filter_variables(self, valid: Tensor) -> "OpticalData":
        return self.replace(
            rays_base = filter_optional_tensor(self.rays_base, valid),
            rays_object = filter_optional_tensor(self.rays_object, valid),
            rays_wavelength = filter_optional_tensor(self.rays_wavelength, valid),
        )


def default_input(
    sampling: dict[str, Any],
    dim: int,
    dtype: torch.dtype = torch.float64,
) -> OpticalData:
    return OpticalData(
        dim=dim,
        dtype=dtype,
        sampling=init_sampling(sampling),
        transforms=[IdentityTransform(dim, dtype)],
        P=torch.empty((0, dim), dtype=dtype),
        V=torch.empty((0, dim), dtype=dtype),
        rays_base=None,
        rays_object=None,
        rays_image=None,
        rays_wavelength=None,
        var_base=None,
        var_object=None,
        var_wavelength=None,
        material=get_material_model("vacuum"),
        loss=torch.tensor(0.0, dtype=dtype),
    )
