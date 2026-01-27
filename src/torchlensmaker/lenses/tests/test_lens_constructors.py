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


def check_lens(lens: tlm.Lens) -> None:
    optics = tlm.Sequential(
        tlm.ObjectAtInfinity(beam_diameter=10, angular_size=0.5, wavelength=(400, 800)),
        tlm.Gap(1),
        lens,
    )

    outputs = optics(tlm.default_input(dim=2))
    scene = tlm.render_sequence(optics, dim=2)


def test_cemented() -> None:
    lens1 = tlm.lenses.cemented(
        surfaces=[
            tlm.Sphere(diameter=30, R=55),
            tlm.Sphere(diameter=30, R=55),
        ],
        gaps=[
            tlm.InnerGap(1.0),
        ],
        materials=["BK7", "air"],
    )

    check_lens(lens1)


def test_singlet() -> None:
    lens1 = tlm.lenses.singlet(
        tlm.Sphere(diameter=30, R=55),
        tlm.InnerGap(1.0),
        tlm.Sphere(diameter=30, R=55),
        material="BK7",
    )

    check_lens(lens1)
