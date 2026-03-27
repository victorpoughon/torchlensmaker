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

from dataclasses import dataclass, replace
from typing import Any, Self

import torch

from torchlensmaker.types import (
    BatchNDTensor,
    BatchTensor,
    IndexTensor,
    MaskTensor,
)


@dataclass
class RayBundle:
    """
    A bundle of parametric light rays in either 2D or 3D
    All rays are in the same medium.
    """

    P: BatchNDTensor
    V: BatchNDTensor
    pupil: BatchNDTensor
    field: BatchNDTensor
    wavel: BatchTensor
    pupil_idx: IndexTensor
    field_idx: IndexTensor
    wavel_idx: IndexTensor
    source_idx: IndexTensor

    @classmethod
    def create(
        cls,
        P: BatchNDTensor,
        V: BatchNDTensor,
        pupil: BatchNDTensor,
        field: BatchNDTensor,
        wavel: BatchTensor,
        pupil_idx: IndexTensor,
        field_idx: IndexTensor,
        wavel_idx: IndexTensor,
        source_idx: IndexTensor,
    ) -> Self:
        float_dtype = P.dtype
        assert float_dtype == V.dtype
        assert float_dtype == pupil.dtype
        assert float_dtype == field.dtype
        assert float_dtype == wavel.dtype
        assert pupil_idx.dtype == torch.int64
        assert field_idx.dtype == torch.int64
        assert wavel_idx.dtype == torch.int64
        assert source_idx.dtype == torch.int64

        device = P.device
        assert device == V.device
        assert device == pupil.device
        assert device == field.device
        assert device == wavel.device
        assert device == pupil_idx.device
        assert device == field_idx.device
        assert device == wavel_idx.device

        return cls(
            P,
            V,
            pupil,
            field,
            wavel,
            pupil_idx,
            field_idx,
            wavel_idx,
            source_idx,
        )

    @classmethod
    def empty(cls, dim: int, dtype: torch.dtype, device: torch.device) -> Self:
        coords_shape = (0,) if dim == 2 else (0, 2)
        return cls.create(
            P=torch.empty((0, dim), dtype=dtype),
            V=torch.empty((0, dim), dtype=dtype),
            pupil=torch.empty(coords_shape, dtype=dtype),
            field=torch.empty(coords_shape, dtype=dtype),
            wavel=torch.empty((0,), dtype=dtype),
            pupil_idx=torch.empty((0,), dtype=torch.int64),
            field_idx=torch.empty((0,), dtype=torch.int64),
            wavel_idx=torch.empty((0,), dtype=torch.int64),
            source_idx=torch.empty((0,), dtype=torch.int64),
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.P.dtype

    @property
    def device(self) -> torch.device:
        return self.P.device

    @property
    def batch_size(self) -> torch.Size:
        return self.P.shape[:-1]

    def replace(self, /, **changes: Any) -> Self:
        return replace(self, **changes)

    def mask(self, valid: MaskTensor) -> Self:
        return type(self)(
            P=self.P[valid],
            V=self.V[valid],
            pupil=self.pupil[valid],
            field=self.field[valid],
            wavel=self.wavel[valid],
            pupil_idx=self.pupil_idx[valid],
            field_idx=self.field_idx[valid],
            wavel_idx=self.wavel_idx[valid],
            source_idx=self.source_idx[valid],
        )

    def cat(self, other: Self) -> Self:
        return type(self)(
            P=torch.cat((self.P, other.P)),
            V=torch.cat((self.V, other.V)),
            pupil=torch.cat((self.pupil, other.pupil)),
            field=torch.cat((self.field, other.field)),
            wavel=torch.cat((self.wavel, other.wavel)),
            pupil_idx=torch.cat((self.pupil_idx, other.pupil_idx)),
            field_idx=torch.cat((self.field_idx, other.field_idx)),
            wavel_idx=torch.cat((self.wavel_idx, other.wavel_idx)),
            source_idx=torch.cat((self.source_idx, other.source_idx)),
        )

    def points_at(self, t: BatchTensor) -> BatchNDTensor:
        "Points on rays at parametric distance t"
        return self.P + t.unsqueeze(-1) * self.V
