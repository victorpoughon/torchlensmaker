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

from typing import Optional, Any, Literal
import torch
import torch.nn as nn

import torchlensmaker as tlm


def cemented(
    surfaces: list[tlm.LocalSurface],
    gaps: list[tlm.PositionGap],
    materials=list[tlm.MaterialModel | str],
) -> tlm.Lens:
    """
    Utility constructor for a cemented lens with N independent surfaces

    Args:
        surfaces: list of surfaces
        gaps: list of position gaps
        materials: list of material, including the exit material

    Returns:
        A lens element
    """

    if len(surfaces) < 2:
        raise RuntimeError(
            f"cemented() expects at least 2 surfaces, got {len(surfaces)}"
        )

    if len(surfaces) - 1 != len(gaps):
        raise RuntimeError(
            f"cemented() expects N-1 gaps for N surfaces, got {len(gaps)} gaps and {len(surfaces)} surfaces"
        )

    if len(surfaces) != len(materials):
        raise RuntimeError(
            f"cemented() expects N materials for N surfaces, got f{len(materials)} materials and {len(surfaces)}"
        )

    # Process position gaps into surface anchors by adding 'origin' at start and end
    pgap_anchors = map(tlm.position_gap_to_anchors, gaps)
    flat = [s for a, b in pgap_anchors for s in (a, b)]
    flat_with_origin = ["origin", *flat, "origin"]
    all_anchors = list(zip(flat_with_origin[::2], flat_with_origin[1::2]))

    print(all_anchors)

    # Add a dummy gap for the loop iteration
    gaps.append(tlm.InnerGap(0))

    assert len(all_anchors) == len(materials) == len(surfaces) == len(gaps)

    sequence: list[nn.Module] = []
    for i, (surface, anchors, gap, material) in enumerate(
        zip(surfaces, all_anchors, gaps, materials)
    ):
        sequence.append(
            tlm.RefractiveSurface(
                surface,
                material=material,
                anchors=anchors,
            )
        )
        if i != len(surfaces) - 1:
            sequence.append(tlm.Gap(gap.gap))

    return tlm.Lens(*sequence)


def singlet(
    surface1: tlm.LocalSurface,
    gap: tlm.PositionGap,
    surface2: tlm.LocalSurface,
    material: tlm.MaterialModel | str,
    exit_material: tlm.MaterialModel | str = "air",
) -> tlm.Lens:
    """
    Utility constructor for a singlet lens with two independent surfaces

    Args:
        surface1: First surface
        gap: position gap between the surfaces
        surface2: Second surface
        material: material of the lens
        exit_material (optional): the material after the lens (default "air")

    Returns:
        A lens element
    """
    return cemented(
        surfaces=[surface1, surface2], gaps=[gap], materials=[material, exit_material]
    )


def symmetric_singlet(
    surface: tlm.LocalSurface,
    gap: tlm.PositionGap,
    material: tlm.MaterialModel | str,
    exit_material: tlm.MaterialModel | str = "air",
) -> tlm.Lens:
    """
    Utility constructor for a symmatric singlet lens with two mirrored surfaces

    Args:
        surface: Lens surface
        gap: position gap between the surfaces
        material: material of the lens
        exit_material (optional): the material after the lens (default "air")

    Returns:
        A lens element
    """
    gap_anchors = tlm.position_gap_to_anchors(gap)

    return tlm.Lens(
        tlm.RefractiveSurface(
            surface, anchors=("origin", gap_anchors[0]), material=material
        ),
        tlm.Gap(gap.gap),
        tlm.RefractiveSurface(
            surface,
            anchors=("origin", gap_anchors[0]),
            scale=-1,
            material=exit_material,
        ),
    )


def doublet(
    surface1: tlm.LocalSurface,
    gap1: tlm.PositionGap,
    surface2: tlm.LocalSurface,
    gap2: tlm.PositionGap,
    surface3: tlm.LocalSurface,
    materials: list[tlm.MaterialModel | str],
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
            list of two or three materials of the lens. The third material indicates
            the exit material, defaults to "air".

    Returns:
        A lens element
    """
    if len(materials) not in (2, 3):
        raise RuntimeError(f"doublet() expects 2 or 3 materials, got {len(materials)}")

    if len(materials) == 2:
        materials.append("air")

    return cemented(
        surfaces=[surface1, surface2, surface3],
        gaps=[gap1, gap2],
        materials=materials,
    )
