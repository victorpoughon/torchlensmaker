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
from torchlensmaker.elements.kinematics import KinematicElement
from torchlensmaker.elements.sequential import Sequential, SubChain
from torchlensmaker.elements.optical_surfaces import (
    CollisionSurface,
    ReflectiveSurface,
    RefractiveSurface,
    Aperture,
    ImagePlane,
)
from torchlensmaker.elements.focal_point import FocalPoint
from torchlensmaker.lenses import LensBase

# from torchlensmaker.lenses import LensBase
from torchlensmaker.core.full_forward import forward_tree
from torchlensmaker.elements.light_sources import LightSourceBase, Wavelength


from .rendering import Collective, RayVariables
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
    LensBase: [LensArtist()],
    SubChain: [ForwardArtist(lambda mod: mod._sequential)],
    CollisionSurface: [CollisionSurfaceArtist()],
    ReflectiveSurface: [ForwardArtist(lambda mod: mod.collision_surface)],
    RefractiveSurface: [RefractiveSurfaceArtist()],
    Aperture: [ForwardArtist(lambda mod: mod.collision_surface)],
    ImagePlane: [ForwardArtist(lambda mod: mod.collision_surface)],
    KinematicElement: [KinematicArtist()],
}


# TODO rename
def render_sequence(
    optics: nn.Module,
    dim: int,
    dtype: torch.dtype,
    sampling: dict[str, Any],
    end: Optional[float] = None,
    title: str = "",
    extra_artists: Dict[type, Artist] = {},
) -> Any:
    # Evaluate the model with forward_tree to keep all intermediate outputs
    input_tree, output_tree = forward_tree(optics, default_input(sampling, dim, dtype))

    # Figure out available ray variables and their range, this will be used for coloring info by tlmviewer
    light_sources_outputs = [
        output
        for mod, output in output_tree.items()
        if isinstance(mod, (LightSourceBase, Wavelength))
    ]
    ray_variables = RayVariables.from_optical_data(light_sources_outputs)

    # Initialize the artist collective
    collective = Collective(
        {**default_artists, **extra_artists}, ray_variables, input_tree, output_tree
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


def default_sampling(
    optics: nn.Module,
    dim: int,
    dtype: torch.dtype = torch.float64,
) -> dict[str, Any]:
    "Default sampling values"

    # TODO could be improved by looking at stack content, etc.
    return {"base": 10, "object": 5, "wavelength": 3}


def show(
    optics: nn.Module,
    sampling: Optional[Dict[str, Any]] = None,
    dim: int = 2,
    dtype: torch.dtype = torch.float64,
    end: Optional[float] = None,
    title: str = "",
    ndigits: int | None = 8,
    controls: object | None = None,
    return_scene: bool = False,
    extra_artists: Dict[type, Artist] = {},
) -> None | Any:
    "Render an optical stack and show it with ipython display"

    if sampling is None:
        sampling = default_sampling(optics, dim, dtype)

    scene = render_sequence(optics, dim, dtype, sampling, end, title, extra_artists)

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
    dtype: torch.dtype = torch.float64,
    sampling: Optional[Dict[str, Any]] = None,
    end: Optional[float] = None,
    title: str = "",
    ndigits: int | None = 8,
    controls: object | None = None,
) -> None:
    "Render and export an optical stack to a tlmviewer json file"

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
