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

from typing import Type

import torch
import torch.nn as nn

from torchlensmaker.sequential.sequential_data import SequentialData


class Marker(nn.Module):
    "WIP"

    def __init__(self, text: str):
        super().__init__()
        self.text = text

    def forward(self, inputs: SequentialData) -> SequentialData:
        return inputs


class Debug(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def sequential(self, data):
        self.func(data)
        return data


def get_elements_by_type(module: nn.Module, typ: Type[nn.Module]) -> nn.ModuleList:
    """
    Returns a ModuleList containing all submodules (including the root module,
    if it matches 'typ') that match the type via isinstance().
    """

    return nn.ModuleList(mod for mod in module.modules() if isinstance(mod, typ))
