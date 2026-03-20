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
from dataclasses import dataclass, field
from typing import Any, Self

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
    input_rays: OrderedDict[str, RayBundle] = field(default_factory=OrderedDict)
    output_rays: OrderedDict[str, RayBundle] = field(default_factory=OrderedDict)
    collisions: OrderedDict[str, tuple[BatchTensor, BatchNDTensor, MaskTensor]] = field(
        default_factory=OrderedDict
    )
    input_joints: OrderedDict[str, Tf] = field(default_factory=OrderedDict)
    output_joints: OrderedDict[str, Tf] = field(default_factory=OrderedDict)
    surfaces: OrderedDict[str, tuple[Tf, SurfaceElement]] = field(
        default_factory=OrderedDict
    )

    @classmethod
    def empty(cls, dim: int) -> Self:
        return cls(dim=dim)

    def add_input_rays(self, key: str, rays: RayBundle) -> None:
        self.input_rays[key] = rays

    def add_output_rays(self, key: str, rays: RayBundle) -> None:
        self.output_rays[key] = rays

    def add_collision(
        self, key: str, t: BatchTensor, normals: BatchNDTensor, valid: MaskTensor
    ) -> None:
        self.collisions[key] = (t, normals, valid)

    def add_input_joint(self, key: str, tf: Tf) -> None:
        self.input_joints[key] = tf

    def add_output_joint(self, key: str, tf: Tf) -> None:
        self.output_joints[key] = tf

    def add_surface(self, key: str, tf_and_surface: tuple[Tf, SurfaceElement]) -> None:
        self.surfaces[key] = tf_and_surface

    def add_scene(self, key: str, other: Self) -> None:
        def merge(this: OrderedDict[str, Any], other: OrderedDict[str, Any]):
            for k, v in other.items():
                this[key + "." + k] = v

        merge(self.output_rays, other.output_rays)
        merge(self.input_joints, other.input_joints)
        merge(self.output_joints, other.output_joints)
        merge(self.surfaces, other.surfaces)
