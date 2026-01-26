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

import math
import torch
import torch.nn as nn
from torchlensmaker.elements.sequential import Dim
from torchlensmaker.optical_data import OpticalData
from typing import Type, Sequence, cast

from torchlensmaker.light_sources.light_sources_elements import GenericLightSource
from torchlensmaker.sampling.sampler_elements import (
    LinspaceSampler1D,
    ExactSampler1D,
    DiskSampler2D,
)


class Marker(nn.Module):
    "WIP"

    def __init__(self, text: str):
        super().__init__()
        self.text = text

    def forward(self, inputs: OpticalData) -> OpticalData:
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


def get_light_source2d(module: nn.Module) -> GenericLightSource:
    """
    Return the unique 2D light source of an optical system

    Raises an exception if there are zero or more than one
    """

    # Get light sources and filter by dim
    sources = get_elements_by_type(module, GenericLightSource)
    sources2d = [s for s in sources if s.dim() == Dim.TWO]

    if len(sources2d) != 1:
        raise RuntimeError(
            f"Expected exactly one 2D light source, got {len(sources2d)}"
        )

    return cast(GenericLightSource, sources2d[0])


def get_light_source3d(module: nn.Module) -> GenericLightSource:
    """
    Return the unique 3D light source of an optical system

    Raises an exception if there are zero or more than one
    """

    # Get light sources and filter by dim
    sources = get_elements_by_type(module, GenericLightSource)
    sources3d = [s for s in sources if s.dim() == Dim.THREE]

    if len(sources3d) != 1:
        raise RuntimeError(
            f"Expected exactly one 3D light source, got {len(sources3d)}"
        )

    return cast(GenericLightSource, sources3d[0])


def init_sampler1d(value: int | Sequence[float]) -> nn.Module:
    if isinstance(value, int):
        return LinspaceSampler1D(value)
    elif isinstance(value, (list, tuple)):
        # TODO support tensor argument to handle dtype and device
        # TODO gpu
        return ExactSampler1D(
            torch.tensor(value, dtype=torch.float64, device=torch.device("cpu"))
        )
    raise RuntimeError(
        f"Sampling: expected number or list of numbers, got {type(value)}: {value}"
    )


def set_sampling2d(
    optics: nn.Module,
    pupil: int | Sequence[float] | None,
    field: int | Sequence[float] | None,
    wavelength: int | Sequence[float] | None,
) -> None:
    source = get_light_source2d(optics)

    if pupil:
        source.sampler_pupil = init_sampler1d(pupil)

    if field:
        source.sampler_field = init_sampler1d(field)

    if wavelength:
        source.sampler_wavelength = init_sampler1d(wavelength)


def init_sampler2d(value: int | Sequence[float]) -> nn.Module:
    if isinstance(value, (float, int)):
        # TODO support two dimensional settings here
        n = math.ceil(math.sqrt(value))
        return DiskSampler2D(n, n)
    elif isinstance(value, (list, tuple)):
        raise NotImplementedError("todo")
    else:
        raise RuntimeError(
            f"Sampling: expected number or list of numbers, got {type(value)}: {value}"
        )


def set_sampling3d(
    optics: nn.Module,
    pupil: int | Sequence[float] | None,
    field: int | Sequence[float] | None,
    wavelength: int | Sequence[float] | None,
) -> None:
    source = get_light_source3d(optics)

    if pupil:
        source.sampler_pupil = init_sampler2d(pupil)

    if field:
        source.sampler_field = init_sampler2d(field)

    if wavelength:
        source.sampler_wavelength = init_sampler1d(wavelength)
