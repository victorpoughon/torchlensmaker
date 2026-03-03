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


def test_paraxial1() -> None:
    doublet = tlm.Lens(
        tlm.RefractiveSurface(
            tlm.SphereByCurvature(4.0, C=0.135327),
            material=tlm.NonDispersiveMaterial(1.517),
        ),
        tlm.Gap(1.05),
        tlm.RefractiveSurface(
            tlm.SphereByCurvature(3.8, C=-0.19311),
            material=tlm.NonDispersiveMaterial(1.649),
        ),
        tlm.Gap(0.4),
        tlm.RefractiveSurface(tlm.SphereByCurvature(4.0, C=-0.06164), material="air"),
    )

    # Paraxial points
    principal_point = tlm.paraxial.rear_principal_point(doublet, wavelength=550)
    focal_point = tlm.paraxial.rear_focal_point(doublet, wavelength=550, h=0.01)
    focal_length = focal_point - principal_point

    print("Lens minimal diameter:", doublet.minimal_diameter().detach().item())
    print("Lens inner thickness:", doublet.inner_thickness().detach().item())
    print("Lens outer thickness:", doublet.outer_thickness().detach().item())
    print("Lens rear principal point:", principal_point.detach().item())
    print("Lens rear focal point:", focal_point.detach().item())
    print("Lens focal length:", focal_length.detach().item())
