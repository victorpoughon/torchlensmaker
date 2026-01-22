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
from ..get_material_model import default_material_models


def test_refractive_index() -> None:
    # refractive index in the visible band should always be >1.0 and < 10.0

    W = torch.linspace(400, 800, 10)
    for name, model in default_material_models.items():
        print(name)
        idx = model(W)
        print(idx)
        assert torch.all(idx >= 1.0)
        assert torch.all(idx <= 10.0)
