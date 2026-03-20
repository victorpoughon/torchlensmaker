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


from collections import OrderedDict
from dataclasses import dataclass
from typing import Self

from torchlensmaker.core.ray_bundle import RayBundle
from torchlensmaker.surfaces.surface_element import SurfaceElement
from torchlensmaker.types import BatchNDTensor, BatchTensor, MaskTensor, Tf


@dataclass
class ModelTrace:
    """
    A concrete realization of a model after sampling and forward evaluation,
    including intermediate data in the model sequence
    """

    dim: int
    rays: OrderedDict[str, RayBundle]
    collisions: OrderedDict[str, tuple[BatchTensor, BatchNDTensor, MaskTensor]]
    joints: OrderedDict[str, Tf]
    surfaces: OrderedDict[str, tuple[Tf, SurfaceElement]]

    @classmethod
    def empty(cls, dim: int) -> Self:
        return cls(
            dim=dim,
            rays=OrderedDict(),
            collisions=OrderedDict(),
            joints=OrderedDict(),
            surfaces=OrderedDict(),
        )

    def add_rays(self, key: str, rays: RayBundle) -> None:
        self.rays[key] = rays

    def add_collision(
        self, key: str, t: BatchTensor, normals: BatchNDTensor, valid: MaskTensor
    ) -> None:
        self.collisions[key] = (t, normals, valid)

    def add_joint(self, key: str, tf: Tf) -> None:
        self.joints[key] = tf

    def add_surface(self, key: str, tf: Tf, surface: SurfaceElement) -> None:
        self.surfaces[key] = (tf, surface)

    def add_scene(self, key: str, other: Self) -> None:
        for k, v in other.rays.items():
            self.add_rays(key + "." + k, v)

        for k, v in other.joints.items():
            self.add_joint(key + "." + k, v)

        for k, v in other.surfaces.items():
            self.add_surface(key + "." + k, *v)
