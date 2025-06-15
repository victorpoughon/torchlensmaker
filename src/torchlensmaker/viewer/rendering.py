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

from itertools import chain
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Optional, TypeAlias, Iterable

from torchlensmaker.core.transforms import forward_kinematic
from torchlensmaker.core.tensor_manip import filter_optional_mask
from torchlensmaker.optical_data import OpticalData

Tensor: TypeAlias = torch.Tensor

LAYER_VALID_RAYS = 1
LAYER_BLOCKED_RAYS = 2
LAYER_OUTPUT_RAYS = 3
LAYER_JOINTS = 4


class Artist:
    def render_module(self, collective: "Collective", module: nn.Module) -> list[Any]:
        raise NotImplementedError

    def render_rays(self, collective: "Collective", module: nn.Module) -> list[Any]:
        raise NotImplementedError


@dataclass
class RayVariables:
    "Available ray variables and their min/max domain"

    variables: list[str]
    domain: dict[str, list[float]]

    @classmethod
    def from_optical_data(cls, optical_data: Iterable[OpticalData]) -> "RayVariables":
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
    data: OpticalData, variables: list[str], valid: Optional[Tensor] = None
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


@dataclass
class Collective:
    "Group of artists"

    artists: Dict[type, Artist]
    ray_variables: RayVariables
    input_tree: dict[nn.Module, Any]
    output_tree: dict[nn.Module, Any]

    def match_artists(self, module: nn.Module) -> list[Artist]:
        "Match an artist to a module"

        artists: list[Artist] = [
            a for typ, a in self.artists.items() if isinstance(module, typ)
        ]

        return [] if len(artists) == 0 else artists[0]

    def render_module(self, module: nn.Module) -> list[Any]:
        artists = self.match_artists(module)

        if len(artists) == 0:
            return []

        # Let the artists render and flatten their returned lists
        renders = [a.render_module(self, module) for a in artists]
        return list(chain(*renders))

    def render_rays(self, module: nn.Module) -> list[Any]:
        artists = self.match_artists(module)

        if len(artists) == 0:
            return []

        # Let the artists render and flatten their returned lists
        renders = [a.render_rays(self, module) for a in artists]
        return list(chain(*renders))

    def render_joints(self, module: nn.Module) -> list[Any]:
        dim, dtype = (
            self.input_tree[module].transforms[0].dim,
            self.input_tree[module].transforms[0].dtype,
        )

        # Final transform list
        tflist = self.output_tree[module].transforms

        points = []

        for i in range(len(tflist)):
            tf = forward_kinematic(tflist[: i + 1])
            joint = tf.direct_points(torch.zeros((dim,), dtype=dtype))

            points.append(joint.tolist())

        return [{"type": "points", "data": points, "layers": [LAYER_JOINTS]}]
