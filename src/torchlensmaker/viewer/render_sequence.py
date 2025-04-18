# This file is part of Torch Lens Maker
# Copyright (C) 2025 Victor Poughon
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
import torchlensmaker as tlm

from typing import Any, Optional, Dict, Iterable
from collections import defaultdict
from dataclasses import dataclass

from torchlensmaker.core.tensor_manip import filter_optional_mask

from torchlensmaker.analysis.colors import (
    color_valid,
    color_focal_point,
    color_blocked,
)

from . import tlmviewer

import json

Tensor = torch.Tensor


LAYER_VALID_RAYS = 1
LAYER_BLOCKED_RAYS = 2
LAYER_OUTPUT_RAYS = 3
LAYER_JOINTS = 4


@dataclass
class RayVariables:
    "Available ray variables and their min/max domain"

    variables: list[str]
    domain: dict[str, list[float]]

    @classmethod
    def from_optical_data(
        cls, optical_data: Iterable[tlm.OpticalData]
    ) -> "RayVariables":
        variables: set[str] = set()
        domain: defaultdict[str, list[float]] = defaultdict(
            lambda: [float("+inf"), float("-inf")]
        )

        def update(var: Optional[Tensor], name: str) -> None:
            if var is not None:
                variables.add(name)
                if var.numel() > 0 and var.min() < domain[name][0]:
                    domain[name][0] = var.min().item()
                if var.numel() > 0 and var.max() > domain[name][1]:
                    domain[name][1] = var.max().item()

        for inputs in optical_data:
            update(inputs.rays_base, "base")
            update(inputs.rays_object, "object")
            update(inputs.rays_wavelength, "wavelength")

        return cls(list(variables), dict(domain))


def ray_variables_dict(
    data: tlm.OpticalData, variables: list[str], valid: Optional[Tensor] = None
) -> dict[str, Tensor]:
    "Convert ray variables from an OpticalData object to a dict of Tensors"
    d = {}

    def update(tensor: Optional[Tensor], name: str) -> None:
        if tensor is not None:
            d[name] = filter_optional_mask(tensor, valid)

    # TODO no support for 2D colormaps in tlmviewer yet
    # but base and object are 2D variables in 3D
    if data.dim == 2:
        update(data.rays_base, "base")
        update(data.rays_object, "object")
    update(data.rays_wavelength, "wavelength")

    return d


def render_rays_until(
    P: Tensor,
    V: Tensor,
    end: Tensor,
    variables: dict[str, Tensor],
    domain: dict[str, list[float]],
    default_color: str,
    layer: int,
) -> list[Any]:
    "Render rays until an absolute X coordinate"
    assert end.dim() == 0
    # div by zero here for vertical rays
    t = (end - P[:, 0]) / V[:, 0]
    ends = P + t.unsqueeze(1).expand_as(V) * V
    return [
        tlmviewer.render_rays(
            P,
            ends,
            variables=variables,
            domain=domain,
            default_color=default_color,
            layer=layer,
        )
    ]


def render_rays_length(
    P: Tensor,
    V: Tensor,
    length: float | Tensor,
    variables: dict[str, Tensor],
    domain: dict[str, list[float]],
    layer: int,
    default_color: str = color_valid,
) -> list[Any]:
    "Render rays with fixed length"

    if isinstance(length, Tensor):
        assert length.dim() in {0, 1}

    if isinstance(length, Tensor) and length.dim() == 1:
        length = length.unsqueeze(1).expand_as(V)

    return [
        tlmviewer.render_rays(
            P,
            P + length * V,
            variables=variables,
            domain=domain,
            default_color=default_color,
            layer=layer,
        )
    ]


class Artist:
    def render_module(self, collective: "Collective", module: nn.Module) -> list[Any]:
        raise NotImplementedError

    def render_rays(self, collective: "Collective", module: nn.Module) -> list[Any]:
        raise NotImplementedError


@dataclass
class Collective:
    "Group of artists"

    artists: Dict[type, Artist]
    ray_variables: RayVariables
    input_tree: dict[nn.Module, tlm.OpticalData]
    output_tree: dict[nn.Module, tlm.OpticalData]

    def match_artist(self, module: nn.Module) -> Optional[Artist]:
        "Match an artist to a module"

        artists: list[Artist] = [
            a for typ, a in self.artists.items() if isinstance(module, typ)
        ]

        return None if len(artists) == 0 else artists[0]

    def render_module(self, module: nn.Module) -> list[Any]:
        artist = self.match_artist(module)

        if artist is None:
            return []

        # Let the artist render the module
        return artist.render_module(
            self,
            module,
        )

    def render_rays(self, module: nn.Module) -> list[Any]:
        artist = self.match_artist(module)

        if artist is None:
            return []

        # Let the artist render the rays
        return artist.render_rays(
            self,
            module,
        )

    def render_joints(self, module: nn.Module) -> list[Any]:
        dim, dtype = (
            self.input_tree[module].transforms[0].dim,
            self.input_tree[module].transforms[0].dtype,
        )

        # Final transform list
        tflist = self.output_tree[module].transforms

        points = []

        for i in range(len(tflist)):
            tf = tlm.forward_kinematic(tflist[: i + 1])
            joint = tf.direct_points(torch.zeros((dim,), dtype=dtype))

            points.append(joint.tolist())

        return [{"type": "points", "data": points, "layers": [LAYER_JOINTS]}]


class KinematicSurfaceArtist(Artist):
    def render_module(self, collective: "Collective", module: nn.Module) -> list[Any]:
        dim, dtype = (
            collective.input_tree[module].transforms[0].dim,
            collective.input_tree[module].transforms[0].dtype,
        )
        chain = collective.input_tree[module].transforms + module.surface_transform(
            dim, dtype
        )
        transform = tlm.forward_kinematic(chain)

        return [tlmviewer.render_surface(module.surface, transform, dim=transform.dim)]

    def render_rays(self, collective: Collective, module: nn.Module) -> list[Any]:
        return collective.render_rays(module.element)


class CollisionSurfaceArtist(Artist):
    def render_module(self, collective: "Collective", module: nn.Module) -> list[Any]:
        points = collective.output_tree[module].P
        normals = collective.output_tree[module].normals

        # return viewer.render_collisions(points, normals)
        return []

    def render_rays(self, collective: Collective, module: nn.Module) -> list[Any]:
        inputs = collective.input_tree[module]
        outputs = collective.output_tree[module]
        # If rays are not blocked, render simply all rays from collision to collision
        if outputs.blocked is None:
            return [
                tlmviewer.render_rays(
                    inputs.P,
                    outputs.P,
                    variables=ray_variables_dict(
                        inputs, collective.ray_variables.variables
                    ),
                    domain=collective.ray_variables.domain,
                    default_color=color_valid,
                    layer=LAYER_VALID_RAYS,
                )
            ]

        # Else, split into colliding and non colliding rays using blocked mask
        else:
            valid = ~outputs.blocked

            group_valid = (
                [
                    tlmviewer.render_rays(
                        inputs.P[valid],
                        outputs.P,
                        variables=ray_variables_dict(
                            inputs, collective.ray_variables.variables, valid
                        ),
                        domain=collective.ray_variables.domain,
                        default_color=color_valid,
                        layer=LAYER_VALID_RAYS,
                    )
                ]
                if inputs.P[valid].numel() > 0
                else []
            )

            P, V = inputs.P[outputs.blocked], inputs.V[outputs.blocked]
            if P.numel() > 0:
                group_blocked = render_rays_until(
                    P,
                    V,
                    inputs.target()[0],
                    variables=ray_variables_dict(
                        inputs, collective.ray_variables.variables, outputs.blocked
                    ),
                    domain=collective.ray_variables.domain,
                    default_color=color_blocked,
                    layer=LAYER_BLOCKED_RAYS,
                )

            else:
                group_blocked = []

            return group_valid + group_blocked


class FocalPointArtist(Artist):
    def render_module(self, collective: "Collective", module: nn.Module) -> list[Any]:
        target = collective.input_tree[module].target().unsqueeze(0)
        return [tlmviewer.render_points(target, color_focal_point)]

    def render_rays(self, collective: Collective, module: nn.Module) -> list[Any]:
        inputs = collective.input_tree[module]

        # Distance from ray origin P to target
        dist = torch.linalg.vector_norm(inputs.P - inputs.target(), dim=1)

        # Always draw rays in their positive t direction
        t = torch.abs(dist)
        return render_rays_length(
            inputs.P,
            inputs.V,
            t,
            layer=LAYER_VALID_RAYS,
            variables=ray_variables_dict(inputs, collective.ray_variables.variables),
            domain=collective.ray_variables.domain,
            default_color=color_valid,
        )


class EndArtist(Artist):
    def __init__(self, end: float):
        self.end = end

    def render_rays(self, collective: Collective, module: nn.Module) -> list[Any]:
        return render_rays_length(
            collective.output_tree[module].P,
            collective.output_tree[module].V,
            self.end,
            variables=ray_variables_dict(
                collective.output_tree[module], collective.ray_variables.variables
            ),
            domain=collective.ray_variables.domain,
            default_color=color_valid,
            layer=LAYER_OUTPUT_RAYS,
        )


class SequentialArtist(Artist):
    def render_module(self, collective: "Collective", module: nn.Module) -> list[Any]:
        nodes = []
        for child in module.children():
            nodes.extend(collective.render_module(child))
        return nodes

    def render_rays(self, collective: Collective, module: nn.Module) -> list[Any]:
        nodes = []
        for child in module.children():
            nodes.extend(collective.render_rays(child))
        return nodes


class LensArtist(Artist):
    def render_module(self, collective: "Collective", module: nn.Module) -> list[Any]:
        nodes = []
        nodes.extend(collective.render_module(module.surface1))
        nodes.extend(collective.render_module(module.surface2))
        return nodes

    def render_rays(self, collective: Collective, module: nn.Module) -> list[Any]:
        nodes = []
        nodes.extend(collective.render_rays(module.surface1))
        nodes.extend(collective.render_rays(module.surface2))
        return nodes


class SubTransformArtist(Artist):
    def render_module(self, collective: "Collective", module: nn.Module) -> list[Any]:
        nodes = []
        nodes.extend(collective.render_module(module.element))
        return nodes

    def render_rays(self, collective: Collective, module: nn.Module) -> list[Any]:
        nodes = []
        nodes.extend(collective.render_rays(module.element))
        return nodes


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


default_artists: Dict[type, Artist] = {
    nn.Sequential: SequentialArtist(),
    tlm.FocalPoint: FocalPointArtist(),
    tlm.LensBase: LensArtist(),
    tlm.Offset: SubTransformArtist(),
    tlm.Rotate: SubTransformArtist(),
    tlm.KinematicSurface: KinematicSurfaceArtist(),
    tlm.CollisionSurface: CollisionSurfaceArtist(),
}


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
    input_tree, output_tree = tlm.forward_tree(
        optics, tlm.default_input(sampling, dim, dtype)
    )

    # Figure out available ray variables and their range, this will be used for coloring info by tlmviewer
    ray_variables = RayVariables.from_optical_data(output_tree.values())

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
