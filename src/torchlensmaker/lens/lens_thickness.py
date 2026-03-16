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


from itertools import islice
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from jaxtyping import Float

from torchlensmaker.core.deep_forward import deep_forward
from torchlensmaker.elements.sequential_data import SequentialData
from torchlensmaker.kinematics.homogeneous_geometry import (
    hom_translate_2d,
    kinematic_chain_append,
    transform_points,
)
from torchlensmaker.optical_surfaces.refractive_surface import RefractiveSurface

if TYPE_CHECKING:
    from .lens import Lens


def lens_inner_thickness(lens: "Lens") -> Float[torch.Tensor, ""]:
    "Thickness of a lens at the center"

    first_surface, last_surface = lens.sequence[0], lens.sequence[-1]

    # Infer dtype, device from the first gap element
    dtype, device = lens.sequence[1].x.dtype, lens.sequence[1].x.device

    # Evaluate the lens with zero rays, so we can extract surface transforms
    with deep_forward(lens) as trace:
        _ = lens(SequentialData.empty(2, dtype))

    front_vertex_tf = trace.outputs[first_surface.surface][3]
    rear_vertex_tf = trace.outputs[last_surface.surface][3]

    root = torch.zeros((2,), dtype=dtype, device=device)
    front_vertex = transform_points(front_vertex_tf.direct, root)
    rear_vertex = transform_points(rear_vertex_tf.direct, root)

    return (rear_vertex - front_vertex)[0]


def lens_outer_thickness(lens: "Lens") -> Float[torch.Tensor, ""]:
    "Thickness of a lens at the outer edge"

    first_surface, last_surface = lens.sequence[0], lens.sequence[-1]
    tau = lens_minimal_diameter(lens) / 2
    front_extent, rear_extent = (
        first_surface.surface.outer_extent(tau),
        last_surface.surface.outer_extent(tau),
    )

    if front_extent is None:
        raise RuntimeError(
            "Lens front surface doesn't have an outer extent defined, cannot compute outer thickness"
        )

    if rear_extent is None:
        raise RuntimeError(
            "Lens rear surface doesn't have an outer extent defined, cannot compute outer thickness"
        )

    # Infer dtype, device from the first gap element
    dtype, device = lens.sequence[1].x.dtype, lens.sequence[1].x.device

    # Evaluate the lens with zero rays, so we can extract surface transforms
    with deep_forward(lens) as trace:
        _ = lens(SequentialData.empty(2, dtype))

    front_vertex_tf = trace.outputs[first_surface.surface][3]
    rear_vertex_tf = trace.outputs[last_surface.surface][3]

    # Append translation along X to include the surface outer edge extent
    zero = torch.zeros((), dtype=dtype, device=device)
    front_extent_tf = hom_translate_2d(torch.stack((front_extent, zero)))
    rear_extent_tf = hom_translate_2d(torch.stack((rear_extent, zero)))

    root = torch.zeros((2,), dtype=dtype)
    front_outer_vertex = transform_points(
        kinematic_chain_append(front_vertex_tf, front_extent_tf).direct, root
    )
    rear_outer_vertex = transform_points(
        kinematic_chain_append(rear_vertex_tf, rear_extent_tf).direct, root
    )

    return (rear_outer_vertex - front_outer_vertex)[0]


def lens_minimal_diameter(lens: "Lens") -> Float[torch.Tensor, ""]:
    """
    Minimal diameter of a lens

    The minimal diameter of a lens is the smallest surface diameter,
    out of all the surfaces in the lens
    """

    mini = lens.sequence[0].surface.diameter
    for mod in lens.sequence:
        if isinstance(mod, RefractiveSurface):
            diam = mod.surface.diameter
            if diam < mini:
                mini = diam

    return mini
