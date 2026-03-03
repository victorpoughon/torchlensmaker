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

from torchlensmaker.core.ray_bundle import RayBundle
from torchlensmaker.types import Tf

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

    # Rays and associated variables
    rays: RayBundle

    # Loss accumulator
    # Tensor of dim 0
    loss: torch.Tensor

    # TODO remove
    def target(self) -> torch.Tensor:
        return transform_points(
            self.fk.direct, torch.zeros((self.dim,), dtype=self.dtype)
        )

    def replace(self, /, **changes: Any) -> "OpticalData":
        return replace(self, **changes)


def default_input(
    dim: int,
    dtype: torch.dtype | None = None,
) -> OpticalData:
    if dtype is None:
        dtype = torch.get_default_dtype()

    tfid = hom_identity(dim, dtype, torch.device("cpu"))  # TODO device support

    rays = RayBundle.create(
        P=torch.empty((0, dim), dtype=dtype),
        V=torch.empty((0, dim), dtype=dtype),
        pupil=torch.empty((0, dim), dtype=dtype),
        field=torch.empty((0, dim), dtype=dtype),
        wavel=torch.empty((0,), dtype=dtype),
        index=torch.empty((0,), dtype=dtype),
        pupil_idx=torch.empty((0,), dtype=torch.int64),
        field_idx=torch.empty((0,), dtype=torch.int64),
        wavel_idx=torch.empty((0,), dtype=torch.int64),
    )

    return OpticalData(
        dim=dim,
        dtype=dtype,
        fk=tfid,
        rays=rays,
        loss=torch.tensor(0.0, dtype=dtype),
    )
