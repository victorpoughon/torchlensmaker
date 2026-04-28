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
from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
from jaxtyping import Float

from torchlensmaker.kinematics.homogeneous_geometry import (
    hom_translate_2d,
    kinematic_chain_append,
    transform_points,
)
from torchlensmaker.optical_surfaces.refractive_surface import RefractiveSurface
from torchlensmaker.sequential.sequential_data import SequentialData
from torchlensmaker.sequential.utils import get_elements_by_type
from torchlensmaker.types import Tf

if TYPE_CHECKING:
    from .lens import Lens


def extract_surface_vertices(
    lens: "Lens", dtype: torch.dtype, device: torch.device
) -> tuple[Tf, Tf]:
    # We assume here that the lens first and last elements are surfaces
    first_surface, lens_core, last_surface = (
        lens.first_surface,
        lens[1:-1],
        lens.last_surface,
    )

    # Evaluate the lens with zero rays, so we can extract surface transforms
    data = SequentialData.empty(2, dtype, device)
    rays, first_surface_outputs = first_surface(data.rays, data.fk)
    front_vertex_tf = first_surface_outputs.tf_surface
    data = lens_core(data.replace(rays=rays, fk=first_surface_outputs.tf_next))
    _, last_surface_outputs = last_surface(data.rays, data.fk)
    rear_vertex_tf = last_surface_outputs.tf_surface

    return front_vertex_tf, rear_vertex_tf


def lens_inner_thickness(lens: "Lens") -> Float[torch.Tensor, ""]:
    "Thickness of a lens at the center"
    # Infer dtype, device from the first gap element
    dtype, device = lens.dtype, lens.device

    # Extract surface transforms for vertices
    front_vertex_tf, rear_vertex_tf = extract_surface_vertices(lens, dtype, device)

    root = torch.zeros((2,), dtype=dtype, device=device)
    front_vertex = transform_points(front_vertex_tf.direct, root)
    rear_vertex = transform_points(rear_vertex_tf.direct, root)

    return (rear_vertex - front_vertex)[0]


def lens_outer_thickness(lens: "Lens") -> Float[torch.Tensor, ""]:
    "Thickness of a lens at the outer edge"

    # We assume here that the lens first and last elements are surfaces
    first_surface, lens_core, last_surface = (
        lens.first_surface,
        lens[1:-1],
        lens.last_surface,
    )

    # Compute extent points of first and last surface
    min_diam = lens_minimal_diameter(lens)
    front_extent, rear_extent = (
        first_surface.surface.outer_extent(min_diam / first_surface.surface.diameter),
        last_surface.surface.outer_extent(min_diam / last_surface.surface.diameter),
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
    dtype, device = lens.dtype, lens.device

    # Extract surface transforms for vertices
    front_vertex_tf, rear_vertex_tf = extract_surface_vertices(lens, dtype, device)

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

    all_diameters: list[float] = []

    for mod in get_elements_by_type(lens, RefractiveSurface):
        diam = mod.surface.diameter
        all_diameters.append(diam)

    return min(all_diameters)
