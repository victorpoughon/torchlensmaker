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

from itertools import chain
from dataclasses import dataclass
from typing import Any, Dict, Optional, TypeAlias

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

    def render_joints(self, collective: "Collective", module: nn.Module) -> list[Any]:
        raise NotImplementedError


def ray_variables_dict(
    data: OpticalData, variables: list[str], valid: Optional[Tensor] = None
) -> dict[str, Tensor]:
    "Convert ray variables from an OpticalData object to a dict of Tensors"
    d = {}

    def update(tensor: Optional[Tensor], name: str) -> None:
        # TODO this if check is temporary to avoid a divide by zero in tlmviewer
        # ideally we would export all three variables allways, and tlmviewer
        # handles correctly degenerate cases like PointSource which has all
        # field coord = 0, or a single wavelength, etc.
        if tensor.numel() > 0 and (tensor.max() - tensor.min()) > 1e-3:
            d[name] = filter_optional_mask(tensor, valid)

    # TODO no support for 2D colormaps in tlmviewer yet
    # but base and object are 2D variables in 3D
    # TODO tlmviewer: rename base/object to pupil/field
    if data.dim == 2:
        update(data.rays_pupil, "base")
        update(data.rays_field, "object")

    update(data.rays_wavelength, "wavelength")

    return d


@dataclass
class Collective:
    "Group of artists"

    artists: Dict[type, Artist]
    ray_variables: list[str]
    ray_variables_domains: dict[str, list[float]]
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
        artists = self.match_artists(module)

        if len(artists) == 0:
            return []

        # Let the artists render and flatten their returned lists
        renders = [a.render_joints(self, module) for a in artists]
        return list(chain(*renders))
