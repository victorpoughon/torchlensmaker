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

from typing import Self
import torch
from tensordict import TensorDict
from torchlensmaker.types import BatchNDTensor, BatchTensor, IndexTensor


class RayBundle(TensorDict):
    """
    A bundle of parametric light rays in either 2D or 3D
    All rays are in the same medium.
    """

    _required_keys = [
        "P",
        "V",
        "pupil",
        "field",
        "wavel",
        "index",
        "pupil_idx",
        "field_idx",
        "wavel_idx",
    ]

    @property
    def P(self) -> BatchNDTensor:
        "Rays origins"
        return self["P"]

    @property
    def V(self) -> BatchNDTensor:
        "Rays directions (unit vectors)"
        return self["V"]

    @property
    def pupil(self) -> BatchNDTensor:
        "Pupil coordinates (in length units)"
        return self["pupil"]

    @property
    def field(self) -> BatchNDTensor:
        "Field coordinates (in length units)"
        return self["field"]

    @property
    def wavel(self) -> BatchTensor:
        "Wavelength (in nanometers)"
        return self["wavel"]

    @property
    def index(self) -> BatchTensor:
        "Index of refraction in the current medium"
        return self["index"]

    @property
    def pupil_idx(self) -> IndexTensor:
        "Index of the rays in the pupil sampling dimension"
        return self["pupil_idx"]

    @property
    def field_idx(self) -> IndexTensor:
        "Index of the rays in the field sampling dimension"
        return self["field_idx"]

    @property
    def wavel_idx(self) -> IndexTensor:
        "Index of the rays in the wavelength sampling dimension"
        return self["wavel_idx"]

    @classmethod
    def create(cls, **kwargs) -> Self:
        missing_keys = [key for key in cls._required_keys if key not in kwargs]
        assert len(missing_keys) == 0, (
            f"RayBundle.create(): required keys missing: {missing_keys}"
        )

        batch_size = kwargs["P"].shape[0]
        self = cls(**kwargs, batch_size=batch_size)

        assert self.pupil_idx.dtype == torch.int64
        assert self.field_idx.dtype == torch.int64
        assert self.wavel_idx.dtype == torch.int64

        return self
