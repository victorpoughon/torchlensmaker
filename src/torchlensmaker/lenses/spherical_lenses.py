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


import torch
from jaxtyping import Float

import torchlensmaker as tlm


def spherical_biconvex(
    diameter: float,
    R: float | Float[torch.Tensor, ""],
    gap: tlm.PositionGap,
    material: tlm.MaterialModel | str,
    exit_material: tlm.MaterialModel | str = "air",
):
    sphere = tlm.SphereByCurvature(diameter, 1/R)

    # TODO check curvature is indeed biconvex TLM-84

    return tlm.lenses.symmetric_singlet(
        sphere,
        gap,
        material,
        exit_material,
    )


def spherical_biconcave(
    diameter: float,
    R: float | Float[torch.Tensor, ""],
    gap: tlm.PositionGap,
    material: tlm.MaterialModel | str,
    exit_material: tlm.MaterialModel | str = "air",
):
    sphere = tlm.SphereByCurvature(diameter, 1/R)

    # TODO check curvature is indeed biconcave TLM-84

    return tlm.lenses.symmetric_singlet(
        sphere,
        gap,
        material,
        exit_material,
    )


def spherical_planoconvex(
    diameter: float,
    R: float | Float[torch.Tensor, ""],
    gap: tlm.PositionGap,
    material: tlm.MaterialModel | str,
    exit_material: tlm.MaterialModel | str = "air",
):
    sphere = tlm.SphereByCurvature(diameter, 1/R)
    plane = tlm.Disk(diameter)

    # TODO check curvature is indeed convex TLM-84

    return tlm.lenses.singlet(
        plane,
        gap,
        sphere,
        material,
        exit_material,
    )


def spherical_convexplano(
    diameter: float,
    R: float | Float[torch.Tensor, ""],
    gap: tlm.PositionGap,
    material: tlm.MaterialModel | str,
    exit_material: tlm.MaterialModel | str = "air",
):
    sphere = tlm.SphereByCurvature(diameter, 1/R)
    plane = tlm.Disk(diameter)

    # TODO check curvature is indeed convex TLM-84

    return tlm.lenses.singlet(
        sphere,
        gap,
        plane,
        material,
        exit_material,
    )
