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

import pytest
import torch
from torchlensmaker.core.ray_bundle import RayBundle


def test_ray_bundle() -> None:
    rays = RayBundle.create(
        P=torch.tensor([[0.0, 0.0]]),
        V=torch.tensor([[1.0, 0.0]]),
        pupil=torch.tensor([[0.0]]),
        field=torch.tensor([[0.0]]),
        wavel=torch.tensor([500.0]),
        pupil_idx=torch.tensor([[0]], dtype=torch.int64),
        field_idx=torch.tensor([[0]], dtype=torch.int64),
        wavel_idx=torch.tensor([0], dtype=torch.int64),
    )

    print(rays)
