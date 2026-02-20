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


from jaxtyping import Float
import torch

from torchlensmaker.types import (
    ScalarTensor,
    Batch2DTensor,
    MaskTensor,
    Tf2D,
    Tf3D,
)

from torchlensmaker.kinematics.homogeneous_geometry import (
    hom_scale_2d,
    hom_translate_2d,
    kinematic_chain_extend_2d,
)

def lens_diameter_domain_2d(
    points: Batch2DTensor, diameter: ScalarTensor
) -> MaskTensor:
    return torch.abs(points[..., 1]) <= diameter / 2


def anchor_transforms_2d(
    anchor0: ScalarTensor,
    anchor1: ScalarTensor,
    scale: ScalarTensor,
    base: Tf2D,
) -> tuple[Tf2D, Tf2D]:
    """
    Compute transforms required to position a surface that has anchors and a scale.

    They are:
    - the "surface transform": the tf that applies to the surface itself to
      position it in global frame
    - the "next transform": the tf that applies to the next element in a
      sequential system where the surface is the latest element of the kinematic
      chain
    """
    # First anchor transform
    t0_x = -scale * anchor0
    t0_y = torch.zeros_like(t0_x)
    tf0 = hom_translate_2d(torch.stack((t0_x, t0_y), dim=-1))

    # Second anchor transform
    t1_x = scale * anchor1
    t1_y = torch.zeros_like(t0_y)
    tf1 = hom_translate_2d(torch.stack((t1_x, t1_y), dim=-1))

    # Compose with the existing kinematic chain
    tfscale = hom_scale_2d(scale)
    tf_surface = kinematic_chain_extend_2d(base, [tf0, tfscale])
    tf_next = kinematic_chain_extend_2d(base, [tf0, tf1])

    return tf_surface, tf_next
