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

from typing import Any, Optional, TypeAlias
from dataclasses import dataclass, replace

import torch
from jaxtyping import Float

from torchlensmaker.core.tensor_manip import filter_optional_tensor
from torchlensmaker.kinematics.homogeneous_geometry import (
    HomMatrix,
    hom_identity,
    transform_points,
)
from torchlensmaker.materials import (
    MaterialModel,
    get_material_model,
)
from torchlensmaker.sampling.samplers import Sampler, init_sampling


@dataclass
class OpticalData:
    # dim is 2 or 3
    # dtype default is torch.float64
    dim: int
    dtype: torch.dtype

    # Sampling configuration for each variable
    sampling: dict[str, Sampler]

    # Forward kinematic chain
    dfk: HomMatrix  # direct
    ifk: HomMatrix  # inverse

    # Light rays in parametric form: P + tV
    P: Float[torch.Tensor, "N D"]
    V: Float[torch.Tensor, "N D"]

    # Light rays wavelength in nm
    rays_wavelength: Float[torch.Tensor, " N"]
    
    # Light rays index of refraction
    rays_index: Float[torch.Tensor, " N"]

    # Rays variables
    # Tensors of shape (N, 2|3) or None
    rays_base: Optional[torch.Tensor]
    rays_object: Optional[torch.Tensor]
    rays_image: Optional[torch.Tensor]

    # Basis of each sampling variable
    # Tensors of shape (*, 2|3)
    # number of rows is the size of each sampling dimension
    var_base: Optional[torch.Tensor]
    var_object: Optional[torch.Tensor]

    # Loss accumulator
    # Tensor of dim 0
    loss: torch.Tensor

    def target(self) -> torch.Tensor:
        return transform_points(self.dfk, torch.zeros((self.dim,), dtype=self.dtype))

    def replace(self, /, **changes: Any) -> "OpticalData":
        return replace(self, **changes)

    def get_rays(self, color_dim: str) -> torch.Tensor:
        if color_dim == "base" and self.rays_base is not None:
            return self.rays_base
        elif color_dim == "object" and self.rays_object is not None:
            return self.rays_object
        elif color_dim == "wavelength" and self.rays_wavelength is not None:
            return self.rays_wavelength
        else:
            raise RuntimeError(f"Unknown or unavailable ray variable '{color_dim}'")

    def filter_variables(self, valid: torch.Tensor) -> "OpticalData":
        return self.replace(
            rays_base=filter_optional_tensor(self.rays_base, valid),
            rays_object=filter_optional_tensor(self.rays_object, valid),
            rays_wavelength=filter_optional_tensor(self.rays_wavelength, valid),
        )


def default_input(
    sampling: dict[str, Any],
    dim: int,
    dtype: torch.dtype | None = None,
) -> OpticalData:
    if dtype is None:
        dtype = torch.get_default_dtype()

    dfk, ifk = hom_identity(dim, dtype, torch.device("cpu"))  # TODO device support

    return OpticalData(
        dim=dim,
        dtype=dtype,
        sampling=init_sampling(sampling),
        dfk=dfk,
        ifk=ifk,
        P=torch.empty((0, dim), dtype=dtype),
        V=torch.empty((0, dim), dtype=dtype),
        rays_wavelength=torch.empty((0,), dtype=dtype),
        rays_index=torch.empty((0,), dtype=dtype),
        rays_base=None,
        rays_object=None,
        rays_image=None,
        var_base=None,
        var_object=None,
        loss=torch.tensor(0.0, dtype=dtype),
    )
