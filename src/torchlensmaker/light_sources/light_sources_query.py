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
from torchlensmaker.core.dim import Dim
from typing import Sequence, cast

from torchlensmaker.sampling.sampler_elements import (
    LinspaceSampler1D,
    ExactSampler1D,
    DiskSampler2D,
)

from torchlensmaker.light_sources.light_sources_elements import GenericLightSource

from torchlensmaker.elements.utils import get_elements_by_type


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
    pupil: int | Sequence[float] | None = None,
    field: int | Sequence[float] | None = None,
    wavel: int | Sequence[float] | None = None,
) -> None:
    # Get light sources and filter by dim
    sources = get_elements_by_type(optics, GenericLightSource)

    if len(sources) == 0:
        return

    if len(sources) > 1:
        raise RuntimeError(f"Expected one light source, got {len(sources)}")

    source = cast(GenericLightSource, sources[0])

    if pupil:
        source.sampler_pupil_2d = init_sampler1d(pupil)

    if field:
        source.sampler_field_2d = init_sampler1d(field)

    if wavel:
        source.sampler_wavelength_2d = init_sampler1d(wavel)


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
    pupil: int | Sequence[float] | None = None,
    field: int | Sequence[float] | None = None,
    wavel: int | Sequence[float] | None = None,
) -> None:
    # Get light sources and filter by dim
    sources = get_elements_by_type(optics, GenericLightSource)

    if len(sources) == 0:
        return

    if len(sources) > 1:
        raise RuntimeError(f"Expected one light source, got {len(sources)}")

    source = cast(GenericLightSource, sources[0])

    if pupil:
        source.sampler_pupil_3d = init_sampler2d(pupil)

    if field:
        source.sampler_field_3d = init_sampler2d(field)

    if wavel:
        source.sampler_wavelength_3d = init_sampler1d(wavel)
