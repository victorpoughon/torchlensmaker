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
from typing import Any, Self, Sequence, Type, TypeVar

import torch
import torch.nn as nn

from torchlensmaker.core.base_module import BaseModule
from torchlensmaker.core.ray_bundle import RayBundle
from torchlensmaker.sequential.focal_point import FocalPoint
from torchlensmaker.sequential.sequential_data import SequentialData
from torchlensmaker.sequential.sequential_element import SequentialElement
from torchlensmaker.sequential.utils import (
    get_elements_by_type,
)
from torchlensmaker.kinematics.kinematics_elements import KinematicElement
from torchlensmaker.light_sources.light_sources_elements import LightSourceBase
from torchlensmaker.light_sources.light_sources_query import (
    set_sampling2d,
    set_sampling3d,
)
from torchlensmaker.optical_surfaces.image_plane import ImagePlane
from torchlensmaker.optical_surfaces.optical_surface import OpticalSurfaceElement
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
        for mod in iter(self):
            data = sequential_forward(mod, data)
        return data

    def raytrace(
        self,
        dim: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> SequentialData:
        data = SequentialData.empty(dim, dtype, device)
        return self(data)

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


def sequential_forward(mod: BaseModule, data: SequentialData):
    """
    Call an element forward function with a SequentialData object
    """

    def istype(typ: Any):
        return isinstance(mod, typ)

    if istype(KinematicElement):
        tf = mod(data.fk)
        return data.replace(fk=tf)
    elif istype(LightSourceBase):
        rays = mod(data.fk.direct)
        return data.replace(rays=rays)
    elif istype(OpticalSurfaceElement):
        rays, _, tf_next = mod(data.rays, data.fk)
        return data.replace(rays=rays, fk=tf_next)
    elif istype(SequentialElement):
        return mod(data)
    elif istype(ImagePlane) or istype(FocalPoint):
        # TODO add a LightSink / LightTarget base class
        # In sequential mode, image plane is transparent to rays
        # We compute its outputs but forward the rays bundle unchanged
        _, _ = mod(data.rays, data.fk)
        return data
    else:
        raise RuntimeError(f"Sequential: element type {type(mod)} not supported")
