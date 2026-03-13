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
    ) -> Self:
        float_dtype = P.dtype
        assert float_dtype == V.dtype
        assert float_dtype == pupil.dtype
        assert float_dtype == field.dtype
        assert float_dtype == wavel.dtype
        assert pupil_idx.dtype == torch.int64
        assert field_idx.dtype == torch.int64
        assert wavel_idx.dtype == torch.int64

        device = P.device
        assert device == V.device
        assert device == pupil.device
        assert device == field.device
        assert device == wavel.device
        assert device == pupil_idx.device
        assert device == field_idx.device
        assert device == wavel_idx.device

        return cls(P, V, pupil, field, wavel, pupil_idx, field_idx, wavel_idx)

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
        )

    def points_at(self, t: BatchTensor) -> BatchNDTensor:
        "Points on rays at parametric distance t"
        return self.P + t.unsqueeze(-1) * self.V

    def propagate_absorb(self, t: BatchTensor, valid: MaskTensor) -> Self:
        "Propagate rays by distance t, removing non valid rays"
        collision_points = self.points_at(t)
        return self.mask(valid).replace(P=collision_points[valid])

    def reorient(self, V: BatchNDTensor) -> Self:
        "Reorient rays to a new direction"
        return self.replace(V=V)

    def reorient_absorb(self, V: BatchNDTensor, valid: MaskTensor) -> Self:
        "Reorient rays to a new direction, removing non valid rays"
        return self.mask(valid).replace(V=V[valid])
