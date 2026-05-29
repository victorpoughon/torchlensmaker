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

from dataclasses import dataclass
from typing import Callable, Self

import torch

from torchlensmaker.types import BatchNDTensor, IndexTensor, MaskTensor


@dataclass
class SampledVariable:
    """
    A sampled variable is the realization of one physical ray variable by
    sampling it at fixed points in its domain.

    Example:
        Let's say we sample the wavelength of rays at M different points to
        obtain values (in nm) over the domain [400, 800]. N is the total number
        of rays in a bundle. Note that we have more than 1 ray per wavelength
        sample, because there are other sampling dimensions.

        So, for this example the tensors of the SampledVariable for wavelength will contain:
            - values: a float tensor of shape (N,) containing the wavelength (in nm) of each ray
            - idx: an int64 tensor of shape (N,) containing the indices of the samples in the sampled domain for each ray
            - domain_values: a float tensor of shape (M,) containing the values (in nm) of the samples over the domain
            - domain_idx: an int64 tensor of shape (M,) containing unique the indices of the samples over the domain
    """

    values: BatchNDTensor  # (N, ...) per-ray physical values
    idx: IndexTensor  # (N,) per-ray indices into the domain
    domain_values: BatchNDTensor  # (M, ...) per-axis physical values
    domain_idx: IndexTensor  # (M,) sorted unique int64

    @classmethod
    def create(
        cls,
        values: BatchNDTensor,
        idx: IndexTensor,
        domain_values: BatchNDTensor,
        domain_idx: IndexTensor,
    ) -> Self:
        assert values.dtype == domain_values.dtype
        assert idx.dtype == torch.int64
        assert domain_idx.dtype == torch.int64
        device = values.device
        assert device == idx.device
        assert device == domain_values.device
        assert device == domain_idx.device
        assert values.shape[1:] == domain_values.shape[1:]
        assert idx.shape == values.shape[:1]
        assert domain_idx.shape == domain_values.shape[:1]
        if len(domain_idx) > 1:
            assert (domain_idx[1:] > domain_idx[:-1]).all(), (
                "domain_idx must be sorted and unique"
            )
        return cls(values, idx, domain_values, domain_idx)

    @classmethod
    def empty(
        cls,
        value_shape: tuple[int, ...],
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> Self:
        return cls(
            values=torch.empty((0, *value_shape), dtype=dtype, device=device),
            idx=torch.empty((0,), dtype=torch.int64, device=device),
            domain_values=torch.empty((0, *value_shape), dtype=dtype, device=device),
            domain_idx=torch.empty((0,), dtype=torch.int64, device=device),
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.values.dtype

    @property
    def device(self) -> torch.device:
        return self.values.device

    def filter(self, valid: MaskTensor) -> Self:
        return type(self)(
            values=self.values[valid],
            idx=self.idx[valid],
            domain_values=self.domain_values,
            domain_idx=self.domain_idx,
        )

    def sample(self, idx: IndexTensor) -> Self:
        return type(self)(
            values=self.values[idx],
            idx=self.idx[idx],
            domain_values=self.domain_values,
            domain_idx=self.domain_idx,
        )

    def map(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> Self:
        "Apply a unary function to both values and domain_values, leaving idx tensors unchanged."
        return type(self)(
            values=fn(self.values),
            idx=self.idx,
            domain_values=fn(self.domain_values),
            domain_idx=self.domain_idx,
        )
