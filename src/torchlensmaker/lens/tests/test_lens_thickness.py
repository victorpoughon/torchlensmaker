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
import torchlensmaker as tlm


def test_lens_thickness() -> None:
    lens1 = tlm.lenses.singlet(
        tlm.SphereByCurvature(diameter=30, C=1/55),
        tlm.InnerGap(1.0),
        tlm.SphereByCurvature(diameter=30, C=1/55),
        material="BK7",
    )
    
    # lens1.inner_thickness()
    # lens1.outer_thickness()
    lens1.minimal_diameter()