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
import torch.nn as nn

from typing import Any, Optional, Dict


from torchlensmaker.optical_data import default_input
from torchlensmaker.kinematics.kinematics_elements import KinematicElement
from torchlensmaker.elements.sequential import Sequential, SubChain
from torchlensmaker.elements.optical_surfaces import (
    CollisionSurface,
    ReflectiveSurface,
    RefractiveSurface,
    Aperture,
    ImagePlane,
)
from torchlensmaker.elements.focal_point import FocalPoint
from torchlensmaker.lens.lens import Lens

# from torchlensmaker.lenses import LensBase
from torchlensmaker.core.full_forward import forward_tree
from torchlensmaker.light_sources.light_sources_elements import LightSourceBase
from torchlensmaker.light_sources.light_sources_query import set_sampling2d, set_sampling3d
from torchlensmaker.elements.utils import get_elements_by_type

from .rendering import Collective
from . import tlmviewer
from .rendering import Artist
from .artists import (
    SequentialArtist,
    FocalPointArtist,
    CollisionSurfaceArtist,
    RefractiveSurfaceArtist,
    LensArtist,
    ForwardArtist,
    EndArtist,
    KinematicArtist,
)

import json

Tensor = torch.Tensor


def inspect_stack(execute_list: list[tuple[nn.Module, Any, Any]]) -> None:
    for module, inputs, outputs in execute_list:
        print(type(module))
        print("inputs.transform:")
        for t in inputs.transforms:
            print(t)
        print()
        print("outputs.transform:")
        for t in outputs.transforms:
            print(t)
        print()


default_artists: Dict[type, list[Artist]] = {
    Sequential: [SequentialArtist()],
    FocalPoint: [FocalPointArtist()],
    Lens: [ForwardArtist(lambda mod: mod.sequence)],
    SubChain: [ForwardArtist(lambda mod: mod._sequential)],
    CollisionSurface: [CollisionSurfaceArtist()],
    ReflectiveSurface: [ForwardArtist(lambda mod: mod.collision_surface)],
    RefractiveSurface: [RefractiveSurfaceArtist()],
    Aperture: [ForwardArtist(lambda mod: mod.collision_surface)],
    ImagePlane: [ForwardArtist(lambda mod: mod.collision_surface)],
    KinematicElement: [KinematicArtist()],
}


def get_domain(optics: nn.Module, dim: int) -> dict[str, list[float]]:
    light_sources = get_elements_by_type(optics, LightSourceBase)

    if len(light_sources) == 0:
        return {}

    # TODO handle multiple light sources
    ls = light_sources[0]

    return ls.domain(dim)


# TODO rename
def render_sequence(
    optics: nn.Module,
    dim: int,
    dtype: torch.dtype | None = None,
    end: Optional[float] = None,
    title: str = "",
    extra_artists: Dict[type, Artist] = {},
) -> Any:
    if dtype is None:
        dtype = torch.get_default_dtype()
    
    # Evaluate the model with forward_tree to keep all intermediate outputs
    input_tree, output_tree = forward_tree(optics, default_input(dim, dtype))

    # Figure out available ray variables and their range, this will be used for coloring info by tlmviewer
    ray_variables = ["base", "object", "wavelength"]
    ray_variables_domains = get_domain(optics, dim)

    # Initialize the artist collective
    collective = Collective(
        {**default_artists, **extra_artists},
        ray_variables,
        ray_variables_domains,
        input_tree,
        output_tree,
    )

    # Initialize the scene
    scene = tlmviewer.new_scene("2D" if dim == 2 else "3D")

    # Render the top level module
    scene["data"].extend(collective.render_module(optics))

    # Render rays
    scene["data"].extend(collective.render_rays(optics))

    # Render kinematic chain joints
    scene["data"].extend(collective.render_joints(optics))

    # Render output rays with end argument
    if end is not None:
        scene["data"].extend(EndArtist(end).render_rays(collective, optics))

    if title != "":
        scene["title"] = title

    return scene


def show(
    optics: nn.Module,
    dim: int = 2,
    dtype: torch.dtype | None = None,
    end: Optional[float] = None,
    title: str = "",
    ndigits: int | None = 8,
    controls: object | None = None,
    return_scene: bool = False,
    extra_artists: Dict[type, Artist] = {},
    pupil: Any | None = None,
    field: Any | None = None,
    wavelength: Any | None = None,
) -> None | Any:
    "Render an optical stack and show it with ipython display"

    if dtype is None:
        dtype = torch.get_default_dtype()

    if dim == 2:
        set_sampling2d(optics, pupil, field, wavelength)
    elif dim == 3:
        set_sampling3d(optics, pupil, field, wavelength)

    scene = render_sequence(optics, dim, dtype, end, title, extra_artists)

    if controls is not None:
        scene["controls"] = controls

    tlmviewer.display_scene(scene, ndigits)

    return scene if return_scene else None


def show2d(*args, **kwargs):
    kwargs["dim"] = 2
    return show(*args, **kwargs)


def show3d(*args, **kwargs):
    kwargs["dim"] = 3
    return show(*args, **kwargs)


def export_json(
    optics: nn.Module,
    filename: str,
    dim: int = 2,
    dtype: torch.dtype | None = None,
    sampling: Optional[Dict[str, Any]] = None,
    end: Optional[float] = None,
    title: str = "",
    ndigits: int | None = 8,
    controls: object | None = None,
) -> None:
    "Render and export an optical stack to a tlmviewer json file"

    if dtype is None:
        dtype = torch.get_default_dtype()

    if sampling is None:
        # TODO figure out a better default based on stack content?
        sampling = {"base": 10, "object": 5, "wavelength": 8}

    scene = render_sequence(optics, dim, dtype, sampling, end, title)

    if controls is not None:
        scene["controls"] = controls

    if ndigits is not None:
        scene = tlmviewer.truncate_scene(scene, ndigits)

    with open(filename, "w") as f:
        json.dump(scene, f)
