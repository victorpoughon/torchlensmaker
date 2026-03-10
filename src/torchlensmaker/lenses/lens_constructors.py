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
from typing import Sequence
import torch
import torch.nn as nn

import torchlensmaker as tlm


def cemented(
    surfaces: list[tlm.SurfaceElement],
    gaps: list[tlm.PositionGap],
    materials=list[tlm.MaterialModel | str],
    anchors: tuple[float, float] | Float[torch.Tensor, " 2"] = (0.0, 0.0),
) -> tlm.Lens:
    """
    Utility constructor for a cemented lens with N independent surfaces

    Args:
        surfaces: list of surfaces
        gaps: list of position gaps
        materials: list of material, including the exit material
        anchors: lens anchors

    Returns:
        A lens element
    """

    if len(surfaces) < 2:
        raise RuntimeError(
            f"cemented() expects at least 2 surfaces, got {len(surfaces)}."
        )

    if len(surfaces) - 1 != len(gaps):
        raise RuntimeError(
            f"cemented() expects N-1 gaps for N surfaces, got {len(gaps)} gaps and {len(surfaces)} surfaces"
        )

    if len(surfaces) + 1 != len(materials):
        raise RuntimeError(
            f"cemented() expects N+1 materials for N surfaces, got {len(materials)} materials and {len(surfaces)} surfaces"
        )

    # Process position gaps into surface anchors by adding 'origin' at start and end
    pgap_anchors = [tlm.position_gap_to_anchors(pg) for pg in gaps]
    flat = [s for a, b in pgap_anchors for s in (a, b)]
    flat_with_origin = [anchors[0], *flat, anchors[1]]
    all_anchors = list(zip(flat_with_origin[::2], flat_with_origin[1::2]))

    # Process materials into material pairs
    all_materials = list(zip(materials, materials[1:]))

    # Add a dummy gap for the loop iteration
    gaps.append(tlm.InnerGap(0))

    assert len(all_anchors) == len(all_materials) == len(surfaces) == len(gaps)

    sequence: list[nn.Module] = []
    for i, (surface, anchors, gap, materials_pair) in enumerate(
        zip(surfaces, all_anchors, gaps, all_materials)
    ):
        sequence.append(
            tlm.RefractiveSurface(
                surface.clone(anchors=anchors),
                materials=materials_pair,
            )
        )
        if i != len(surfaces) - 1:
            sequence.append(tlm.Gap(gap.gap))

    return tlm.Lens(*sequence)


def singlet(
    surface1: tlm.SurfaceElement,
    gap: tlm.PositionGap,
    surface2: tlm.SurfaceElement,
    material: tlm.MaterialModel | str,
    entry_material: tlm.MaterialModel | str = "air",
    exit_material: tlm.MaterialModel | str = "air",
    anchors: tuple[float, float] | Float[torch.Tensor, " 2"] = (0.0, 0.0),
) -> tlm.Lens:
    """
    Utility constructor for a singlet lens with two independent surfaces

    Args:
        surface1: First surface
        gap: position gap between the surfaces
        surface2: Second surface
        material: material of the lens
        entry_material (optional): the material before the lens (default "air")
        exit_material (optional): the material after the lens (default "air")

    Returns:
        A lens element
    """
    return cemented(
        surfaces=[surface1, surface2],
        gaps=[gap],
        materials=[entry_material, material, exit_material],
        anchors=anchors,
    )


def symmetric_singlet(
    surface: tlm.SurfaceElement,
    gap: tlm.PositionGap,
    material: tlm.MaterialModel | str,
    entry_material: tlm.MaterialModel | str = "air",
    exit_material: tlm.MaterialModel | str = "air",
    anchors: tuple[float, float] | Float[torch.Tensor, " 2"] = (0.0, 0.0),
) -> tlm.Lens:
    """
    Utility constructor for a symmatric singlet lens with two mirrored surfaces

    Args:
        surface: Lens surface
        gap: position gap between the surfaces
        material: material of the lens
        entry_material (optional): the material before the lens (default "air")
        exit_material (optional): the material after the lens (default "air")

    Returns:
        A lens element
    """
    gap_anchors = tlm.position_gap_to_anchors(gap)

    return tlm.Lens(
        tlm.RefractiveSurface(
            surface.clone(anchors=(anchors[0], gap_anchors[0])),
            materials=(entry_material, material),
        ),
        tlm.Gap(gap.gap),
        tlm.RefractiveSurface(
            surface.clone(
                anchors=(gap_anchors[1], anchors[1]),
                scale=-1,
            ),
            materials=(material, exit_material),
        ),
    )


def semiplanar_rear(
    surface: tlm.SurfaceElement,
    gap: tlm.PositionGap,
    material: tlm.MaterialModel | str,
    entry_material: tlm.MaterialModel | str = "air",
    exit_material: tlm.MaterialModel | str = "air",
    scale: float = 1.0,
    anchors: tuple[float, float] | Float[torch.Tensor, " 2"] = (0.0, 0.0),
) -> tlm.Lens:
    """
    Utility constructor for a semiplanar singlet lens with a surface at the
    front and a plane at the rear.

    Args:
        surface: front lens surface
        gap: position gap between the surfaces
        material: material of the lens
        exit_material (optional): the material after the lens (default "air")

    Returns:
        A lens element
    """
    gap_anchors = tlm.position_gap_to_anchors(gap)

    return tlm.Lens(
        tlm.RefractiveSurface(
            surface.clone(anchors=(anchors[0], gap_anchors[0]), scale=scale),
            materials=(entry_material, material),
        ),
        tlm.Gap(gap.gap),
        tlm.RefractiveSurface(
            tlm.Disk(surface.diameter),
            materials=(material, exit_material),
        ),
    )


def semiplanar_front(
    surface: tlm.SurfaceElement,
    gap: tlm.PositionGap,
    material: tlm.MaterialModel | str,
    entry_material: tlm.MaterialModel | str = "air",
    exit_material: tlm.MaterialModel | str = "air",
    scale: float = 1.0,
    anchors: tuple[float, float] | Float[torch.Tensor, " 2"] = (0.0, 0.0),
) -> tlm.Lens:
    """
    Utility constructor for a semiplanar singlet lens with a plane at the front
    and a surface at the rear.

    Args:
        surface: front lens surface
        gap: position gap between the surfaces
        material: material of the lens
        exit_material (optional): the material after the lens (default "air")

    Returns:
        A lens element
    """
    gap_anchors = tlm.position_gap_to_anchors(gap)

    return tlm.Lens(
        tlm.RefractiveSurface(
            tlm.Disk(surface.diameter), materials=(entry_material, material)
        ),
        tlm.Gap(gap.gap),
        tlm.RefractiveSurface(
            surface.clone(anchors=(gap_anchors[1], anchors[1]), scale=scale),
            materials=(material, exit_material),
        ),
    )


def doublet(
    surface1: tlm.SurfaceElement,
    gap1: tlm.PositionGap,
    surface2: tlm.SurfaceElement,
    gap2: tlm.PositionGap,
    surface3: tlm.SurfaceElement,
    materials: Sequence[tlm.MaterialModel | str],
    anchors: tuple[float, float] | Float[torch.Tensor, " 2"] = (0.0, 0.0),
) -> tlm.Lens:
    """
    Utility constructor for a doublet lens with three independent surfaces

    Args:
        surface1: First surface
        gap1: first position gap
        surface2: Second surface
        gap2: second position gap
        surface3: Third surface
        materials:
            list of two or four materials of the lens. The first and last
            materials indicates the exit material, defaults to "air".

    Returns:
        A lens element
    """
    if len(materials) not in (2, 4):
        raise RuntimeError(f"doublet() expects 2 or 4 materials, got {len(materials)}")

    if len(materials) == 2:
        materials = ["air", *materials, "air"]

    return cemented(
        surfaces=[surface1, surface2, surface3],
        gaps=[gap1, gap2],
        materials=materials,
        anchors=anchors,
    )
