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

import operator
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from itertools import islice
from typing import Any, Callable, Self, Sequence, Type, TypeVar, cast

import torch
import torch.nn as nn

from torchlensmaker.core.base_module import BaseModule
from torchlensmaker.core.ray_bundle import RayBundle
from torchlensmaker.kinematics.kinematics_elements import KinematicElement
from torchlensmaker.light_sources.light_sources_elements import LightSourceBase
from torchlensmaker.light_sources.light_sources_query import (
    set_sampling2d,
    set_sampling3d,
)
from torchlensmaker.light_targets.focal_point import FocalPoint
from torchlensmaker.light_targets.image_plane import ImagePlane
from torchlensmaker.light_targets.light_target import LightTarget, LightTargetOutput
from torchlensmaker.optical_surfaces.optical_surface import OpticalSurfaceElement
from torchlensmaker.sequential.model_trace import ModelTrace
from torchlensmaker.sequential.sequential_data import SequentialData
from torchlensmaker.sequential.sequential_element import SequentialElement
from torchlensmaker.sequential.utils import (
    get_elements_by_type,
)
from torchlensmaker.types import HomMatrix, Tf


class SubChain(SequentialElement):
    def __init__(self, *children: BaseModule):
        super().__init__()
        self._sequential = Sequential(*children)

    def clone(self, **overrides: Any) -> Self:
        return type(self)(*self._sequential)

    def forward(self, data: SequentialData) -> SequentialData:
        output: SequentialData = self._sequential(data)
        return output.replace(fk=data.fk)

    def forward_trace(
        self, data: SequentialData, prefix: str, trace: ModelTrace
    ) -> SequentialData:
        output: SequentialData = self._sequential.forward_trace(data, prefix, trace)
        return output.replace(fk=data.fk)


_V = TypeVar("_V")


class Sequential(SequentialElement):
    def __init__(self, *args: BaseModule):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module.clone())
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module.clone())

    def clone(self, **overrides: Any) -> Self:
        return type(self)(*self)

    def reverse(self) -> Self:
        seq = [mod.reverse() for mod in reversed(self)]
        return type(self)(*seq)

    def _get_item_by_idx(self, iterator: Iterable[_V], idx: int) -> _V:
        """Get the idx-th item of the iterator."""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError(f"index {idx} is out of range")
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx: slice | int) -> BaseModule:
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[BaseModule]:
        return iter(self._modules.values())

    def forward(self, data: SequentialData) -> SequentialData:
        for key, mod in self._modules.items():
            mod = cast(SequentialElement, mod)
            data = sequential_forward(mod, key, data, None)
        return data

    def __call__(self, data: SequentialData) -> SequentialData:
        # this is there only so that type hints work
        return cast(SequentialData, super().__call__(data))

    def forward_trace(
        self, data: SequentialData, prefix: str, trace: ModelTrace
    ) -> SequentialData:
        # iterate on _modules to not skip any duplicated modules
        for key, mod in self._modules.items():
            mod = cast(SequentialElement, mod)
            data = sequential_forward(mod, prefix + key, data, trace)
        return data

    def get_elements_by_type(self, typ: Type[nn.Module]) -> nn.ModuleList:
        return get_elements_by_type(self, typ)

    def set_sampling2d(
        self,
        pupil: int | Sequence[float] | None = None,
        field: int | Sequence[float] | None = None,
        wavel: int | Sequence[float] | None = None,
    ) -> None:
        return set_sampling2d(self, pupil, field, wavel)

    def set_sampling3d(
        self,
        pupil: int | Sequence[float] | None = None,
        field: int | Sequence[float] | None = None,
        wavel: int | Sequence[float] | None = None,
    ) -> None:
        return set_sampling3d(self, pupil, field, wavel)


def sequential_forward(
    mod: BaseModule,
    key: str,
    data: SequentialData,
    trace: ModelTrace | None,
) -> SequentialData:
    """
    Call an element forward function with a SequentialData object, optionally record in an OpticalScene
    """

    if isinstance(mod, KinematicElement):
        tf = mod(data.fk)
        if trace:
            trace.add_input_joint(key, data.fk)
            trace.add_output_joint(key, tf)
        return data.replace(fk=tf)
    elif isinstance(mod, LightSourceBase):
        # Merge this light source rays with any previous rays
        new_rays = data.rays.cat(mod(data.fk.direct))
        if trace:
            trace.add_output_rays(key, new_rays)
        return data.replace(rays=new_rays)
    elif isinstance(mod, OpticalSurfaceElement):
        new_rays, t, normals, valid, tf_surface, tf_next = mod(data.rays, data.fk)
        if trace:
            trace.add_input_joint(key, data.fk)
            trace.add_output_joint(key, tf_next)
            trace.add_input_rays(key, data.rays)
            trace.add_output_rays(key, new_rays)
            trace.add_surface(key, (tf_surface, mod.surface))
            trace.add_collision(key, t, normals, valid)
        return data.replace(rays=new_rays, fk=tf_next)
    elif isinstance(mod, SequentialElement):
        if trace:
            new_data = mod.forward_trace(data, key + ".", trace)
        else:
            new_data = mod(data)
        return new_data
    elif isinstance(mod, ImagePlane):
        out = cast(LightTargetOutput, mod(data.rays, data.fk))

        if trace:
            trace.add_input_joint(key, data.fk)
            trace.add_output_joint(key, out.tf_next)
            trace.add_input_rays(key, data.rays)
            trace.add_surface(key, (out.tf_surface, mod.surface))
            trace.add_collision(key, out.t, out.normals, out.valid)

        # In sequential mode, light targets are transparent to rays
        # We evaluate the optical element outputs but forward the data unchanged
        return data
    elif isinstance(mod, FocalPoint):
        out = mod(data.rays, data.fk)

        if trace:
            trace.add_input_joint(key, data.fk)
            trace.add_output_joint(key, out.tf_next)
            trace.add_input_rays(key, data.rays)
            trace.add_focal_point(key, data.fk)

        # In sequential mode, light targets are transparent to rays
        # We evaluate the optical element outputs but forward the data unchanged
        return data
    else:
        raise RuntimeError(f"Sequential: element type {type(mod)} not supported")
