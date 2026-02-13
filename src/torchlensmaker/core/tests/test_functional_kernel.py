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


import pytest

import torch
from torchlensmaker.core.functional_kernel import (
    kernel_flat_io,
    kernel_flat_names,
)
from torchlensmaker.types import Tf2D, Tf3D
from torchlensmaker.kinematics.homogeneous_geometry import (
    hom_identity_2d,
    hom_identity_3d,
)


def assert_equal_tensors_of_tuples(
    a: tuple[torch.Tensor, ...], b: tuple[torch.Tensor, ...]
):
    assert isinstance(a, tuple)
    assert isinstance(b, tuple)
    assert len(a) == len(b)
    assert [isinstance(e, torch.Tensor) for e in a]
    assert [isinstance(e, torch.Tensor) for e in b]
    assert [torch.all(ea == eb) for ea, eb in zip(a, b)]


def test_flatten_kernel_io() -> None:
    t = torch.tensor
    t2 = hom_identity_2d(torch.get_default_dtype(), torch.get_default_device())
    t3 = hom_identity_3d(torch.get_default_dtype(), torch.get_default_device())

    # single tensor
    assert_equal_tensors_of_tuples(kernel_flat_io(t(1)), (t(1),))

    # tuple of tensors
    assert_equal_tensors_of_tuples(
        kernel_flat_io((t(1), t(2))),
        (
            t(1),
            t(2),
        ),
    )

    # tuple of tuple tensors
    assert_equal_tensors_of_tuples(
        kernel_flat_io((t(1), (t(2), t(3)))),
        (
            t(1),
            t(2),
            t(3),
        ),
    )

    # Tf2D
    assert_equal_tensors_of_tuples(kernel_flat_io(t2), (t2.direct, t2.inverse))

    # Tf3D
    assert_equal_tensors_of_tuples(kernel_flat_io(t3), (t3.direct, t3.inverse))

    # Heavily nested case
    assert_equal_tensors_of_tuples(
        kernel_flat_io((t(1), (t(2), t(3)), (t2, t3), (t(4),))),
        (
            t(1),
            t(2),
            t(3),
            t2.direct,
            t2.inverse,
            t3.direct,
            t3.inverse,
            t(4),
        ),
    )


def test_flatten_kernel_names() -> None:
    t = torch.tensor
    t2 = hom_identity_2d(torch.get_default_dtype(), torch.get_default_device())
    t3 = hom_identity_3d(torch.get_default_dtype(), torch.get_default_device())

    # TF2D
    assert kernel_flat_names([("tf_in", Tf2D)]) == ["tf_in.direct", "tf_in.inverse"]

    # TF3D
    assert kernel_flat_names([("tf_in", Tf3D)]) == ["tf_in.direct", "tf_in.inverse"]

    # mix case
    assert kernel_flat_names([("a", Tf3D), ("b", Tf3D), ("c", t(5))]) == [
        "a.direct",
        "a.inverse",
        "b.direct",
        "b.inverse",
        "c",
    ]
