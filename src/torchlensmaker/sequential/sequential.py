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
from typing import Any, Self, Sequence, Type, TypeVar, cast

import torch
import torch.nn as nn

from torchlensmaker.core.base_module import BaseModule
from torchlensmaker.light_sources.light_sources_query import (
    set_sampling2d,
    set_sampling3d,
)
from torchlensmaker.sequential.sequential_data import SequentialData
from torchlensmaker.sequential.utils import (
    get_elements_by_type,
)


class SubChain(BaseModule):
    def __init__(self, *children: BaseModule):
        super().__init__()
        self._sequential = Sequential(*children)

    def clone(self, **overrides: Any) -> Self:
        return type(self)(*self._sequential)

    def forward(self, data: SequentialData) -> SequentialData:
        output: SequentialData = self._sequential(data)
        return output.replace(fk=data.fk)

    def sequential(self, data: SequentialData) -> SequentialData:
        new_data = self(data)
        return new_data


_V = TypeVar("_V")


class Sequential(BaseModule):
    def __init__(self, *args: BaseModule):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

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
            mod = cast(BaseModule, mod)
            data = mod.sequential(data)
        return data

    def sequential(self, data: SequentialData) -> SequentialData:
        new_data = self(data)
        return new_data

    def __call__(self, data: SequentialData) -> SequentialData:
        # this is there only so that type hints work
        return cast(SequentialData, super().__call__(data))

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
