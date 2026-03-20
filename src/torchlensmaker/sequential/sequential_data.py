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

from dataclasses import dataclass, replace
from typing import Any, Self

import torch

from torchlensmaker.core.ray_bundle import RayBundle
from torchlensmaker.kinematics.homogeneous_geometry import hom_identity
from torchlensmaker.types import Tf


@dataclass
class SequentialData:
    # Rays and associated variables
    rays: RayBundle

    # Forward kinematic chain
    fk: Tf

    def replace(self, /, **changes: Any) -> "SequentialData":
        return replace(self, **changes)

    @classmethod
    def empty(
        cls,
        dim: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> Self:
        if dtype is None:
            dtype = torch.get_default_dtype()
        if device is None:
            device = torch.get_default_device()

        tfid = hom_identity(dim, dtype, device)

        rays = RayBundle.create(
            P=torch.empty((0, dim), dtype=dtype),
            V=torch.empty((0, dim), dtype=dtype),
            pupil=torch.empty((0, dim), dtype=dtype),
            field=torch.empty((0, dim), dtype=dtype),
            wavel=torch.empty((0,), dtype=dtype),
            pupil_idx=torch.empty((0,), dtype=torch.int64),
            field_idx=torch.empty((0,), dtype=torch.int64),
            wavel_idx=torch.empty((0,), dtype=torch.int64),
        )

        return cls(
            fk=tfid,
            rays=rays,
        )
