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

from torchlensmaker.core.sampled_variable import SampledVariable
from torchlensmaker.types import (
    BatchNDTensor,
    BatchTensor,
    MaskTensor,
)


@dataclass
class RayBundle:
    """
    A bundle of parametric light rays in either 2D or 3D.
    All rays are in the same medium.
    """

    P: BatchNDTensor
    V: BatchNDTensor
    pupil: SampledVariable
    field: SampledVariable
    wavel: SampledVariable
    source: SampledVariable

    @classmethod
    def create(
        cls,
        P: BatchNDTensor,
        V: BatchNDTensor,
        pupil: SampledVariable,
        field: SampledVariable,
        wavel: SampledVariable,
        source: SampledVariable,
    ) -> Self:
        float_dtype = P.dtype
        assert float_dtype == V.dtype
        assert float_dtype == pupil.values.dtype
        assert float_dtype == field.values.dtype
        assert float_dtype == wavel.values.dtype
        assert float_dtype == source.values.dtype

        device = P.device
        assert device == V.device
        assert device == pupil.values.device
        assert device == field.values.device
        assert device == wavel.values.device
        assert device == source.values.device

        return cls(P, V, pupil, field, wavel, source)

    @classmethod
    def empty(cls, dim: int, dtype: torch.dtype | None = None, device: torch.device | None = None) -> Self:
        coords_shape: tuple[int, ...] = () if dim == 2 else (2,)
        return cls.create(
            P=torch.empty((0, dim), dtype=dtype, device=device),
            V=torch.empty((0, dim), dtype=dtype, device=device),
            pupil=SampledVariable.empty(coords_shape, dtype, device),
            field=SampledVariable.empty(coords_shape, dtype, device),
            wavel=SampledVariable.empty((), dtype, device),
            source=SampledVariable.empty((), dtype, device),
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
            pupil=self.pupil.mask(valid),
            field=self.field.mask(valid),
            wavel=self.wavel.mask(valid),
            source=self.source.mask(valid),
        )

    def cat(self, other: Self) -> Self:
        return type(self)(
            P=torch.cat((self.P, other.P)),
            V=torch.cat((self.V, other.V)),
            pupil=self.pupil.cat(other.pupil),
            field=self.field.cat(other.field),
            wavel=self.wavel.cat(other.wavel),
            source=self.source.cat(other.source),
        )

    def points_at(self, t: BatchTensor) -> BatchNDTensor:
        "Points on rays at parametric distance t"
        return self.P + t.unsqueeze(-1) * self.V
    
    def _sv_by_name(self, name: str) -> SampledVariable:
        allowed = ["pupil", "field", "wavel", "source"]
        if name not in allowed:
            raise ValueError(f"ray bundle var must be one of {allowed}")
        return getattr(self, name)

    def split_by(
        self,
        name1: str,
        name2: str | None = None,
    ) -> "list[RayBundle] | list[list[RayBundle]]":
        """Partition the bundle by one or two SampledVariable fields.

        Each cell in the returned list (or grid) contains the rays whose
        var1.idx (and var2.idx) match that cell's domain key. Cells where
        all rays were filtered out are empty RayBundles whose domain tensors
        are still intact, so callers can label them from domain_values.
        """

        var1 = self._sv_by_name(name1)

        if name2 is None:
            return [self.mask(var1.idx == k) for k in var1.domain_idx]

        result = []
        for k1 in var1.domain_idx:
            sub = self.mask(var1.idx == k1)
            sub_var2 = sub._sv_by_name(name2)
            result.append([sub.mask(sub_var2.idx == k2) for k2 in sub_var2.domain_idx])
        return result
