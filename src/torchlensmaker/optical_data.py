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

from typing import Any, Optional
from dataclasses import dataclass, replace

import torch
from jaxtyping import Float


from torchlensmaker.types import Tf
from torchlensmaker.core.tensor_manip import filter_optional_tensor, filter_optional_mask
from torchlensmaker.kinematics.homogeneous_geometry import (
    hom_identity,
    transform_points,
)


@dataclass
class OpticalData:
    # dim is 2 or 3
    dim: int
    dtype: torch.dtype

    # Forward kinematic chain
    fk: Tf

    # Light rays in parametric form: P + tV
    P: Float[torch.Tensor, "N D"]
    V: Float[torch.Tensor, "N D"]

    # Ray variables
    # Tensors of shape (N, 2|3)
    rays_wavelength: Float[torch.Tensor, " N"]  # wavelength in nm
    rays_index: Float[torch.Tensor, " N"]  # index of refraction
    rays_pupil: Float[torch.Tensor, "N D"]  # pupil coordinates
    rays_field: Float[torch.Tensor, "N D"]  # field coordinates

    # TODO remove? image plane coordinates
    rays_image: Optional[torch.Tensor]

    # Loss accumulator
    # Tensor of dim 0
    loss: torch.Tensor

    def target(self) -> torch.Tensor:
        return transform_points(self.fk.direct, torch.zeros((self.dim,), dtype=self.dtype))

    def replace(self, /, **changes: Any) -> "OpticalData":
        return replace(self, **changes)

    def get_rays(self, color_dim: str) -> torch.Tensor:
        if color_dim == "base" and self.rays_pupil is not None:
            return self.rays_pupil
        elif color_dim == "object" and self.rays_field is not None:
            return self.rays_field
        elif color_dim == "wavelength" and self.rays_wavelength is not None:
            return self.rays_wavelength
        else:
            raise RuntimeError(f"Unknown or unavailable ray variable '{color_dim}'")

    def filter_variables(self, valid: torch.Tensor) -> "OpticalData":
        return self.replace(
            rays_pupil=filter_optional_tensor(self.rays_pupil, valid),
            rays_field=filter_optional_tensor(self.rays_field, valid),
            rays_wavelength=filter_optional_tensor(self.rays_wavelength, valid),
        )

    def ray_variables_dict(
        self, valid: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        "Convert ray variables from to a dict of Tensors"
        d = {}

        def update(tensor: Optional[torch.Tensor], name: str) -> None:
            # TODO this if check is temporary to avoid a divide by zero in tlmviewer
            # ideally we would export all three variables allways, and tlmviewer
            # handles correctly degenerate cases like PointSource which has all
            # field coord = 0, or a single wavelength, etc.
            if tensor.numel() > 0 and (tensor.max() - tensor.min()) > 1e-3:
                d[name] = filter_optional_mask(tensor, valid)

        # TODO no support for 2D colormaps in tlmviewer yet
        # but base and object are 2D variables in 3D
        # TODO tlmviewer: rename base/object to pupil/field
        if self.dim == 2:
            update(self.rays_pupil, "base")
            update(self.rays_field, "object")

        update(self.rays_wavelength, "wavelength")

        return d


def default_input(
    dim: int,
    dtype: torch.dtype | None = None,
) -> OpticalData:
    if dtype is None:
        dtype = torch.get_default_dtype()

    tfid = hom_identity(dim, dtype, torch.device("cpu"))  # TODO device support

    return OpticalData(
        dim=dim,
        dtype=dtype,
        fk=tfid,
        P=torch.empty((0, dim), dtype=dtype),
        V=torch.empty((0, dim), dtype=dtype),
        rays_wavelength=torch.empty((0,), dtype=dtype),
        rays_index=torch.empty((0,), dtype=dtype),
        rays_pupil=torch.empty((0, dim), dtype=dtype),
        rays_field=torch.empty((0, dim), dtype=dtype),
        rays_image=None,
        loss=torch.tensor(0.0, dtype=dtype),
    )
