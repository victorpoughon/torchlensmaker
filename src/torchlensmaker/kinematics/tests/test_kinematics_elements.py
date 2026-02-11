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
from typing import Any

import torch
import torch.nn as nn

from torchlensmaker.types import HomMatrix2D, HomMatrix3D

from torchlensmaker.kinematics.homogeneous_geometry import (
    hom_identity_2d,
    hom_identity_3d,
)

from torchlensmaker.kinematics.kinematics_elements import (
    Translate2D,
    TranslateVec2D,
    Translate3D,
    TranslateVec3D,
    Rotate2D,
    Rotate3D,
    AbsolutePosition3D,
    AbsolutePosition2D,
    Gap,
    Rotate,
    Translate,
    KinematicSequential,
)

from torchlensmaker.testing.test_utils import (
    check_model_eval,
    check_model_eval_and_grad,
)


def check_valid_kinematic_chain_2d(
    dfk: HomMatrix2D,
    ifk: HomMatrix2D,
    expected_dtype: torch.dtype,
    expected_device: torch.device,
) -> None:
    assert torch.all(torch.isfinite(dfk))
    assert torch.all(torch.isfinite(ifk))
    assert dfk.shape == (3, 3)
    assert ifk.shape == (3, 3)
    assert dfk.dtype == expected_dtype
    assert ifk.dtype == expected_dtype
    assert dfk.device == expected_device
    assert ifk.device == expected_device

    # Inverse x Direct and Direct x Inverse should be close to the identity
    assert torch.allclose(
        dfk @ ifk, torch.eye(3, dtype=expected_dtype, device=expected_device)
    )
    assert torch.allclose(
        ifk @ dfk, torch.eye(3, dtype=expected_dtype, device=expected_device)
    )


def check_valid_kinematic_chain_3d(
    dfk: HomMatrix3D,
    ifk: HomMatrix3D,
    expected_dtype: torch.dtype,
    expected_device: torch.device,
) -> None:
    assert torch.all(torch.isfinite(dfk))
    assert torch.all(torch.isfinite(ifk))
    assert dfk.shape == (4, 4)
    assert ifk.shape == (4, 4)
    assert dfk.dtype == expected_dtype
    assert ifk.dtype == expected_dtype
    assert dfk.device == expected_device
    assert ifk.device == expected_device

    # Inverse x Direct and Direct x Inverse should be close to the identity
    assert torch.allclose(
        dfk @ ifk, torch.eye(4, dtype=expected_dtype, device=expected_device)
    )
    assert torch.allclose(
        ifk @ dfk, torch.eye(4, dtype=expected_dtype, device=expected_device)
    )


def check_kinematic_element_2d(
    element: nn.ModuleList,
    trainable: bool,
    dtype: torch.dtype,
    device: torch.device,
    allow_none_grad: bool = False,
) -> None:
    # Check that kinematic model can be evaluated and differentiated
    dfk, ifk = hom_identity_2d(dtype, device)

    if trainable:
        dfk_out, ifk_out = check_model_eval_and_grad(
            element, (dfk, ifk), allow_none_grad
        )
    else:
        dfk_out, ifk_out = check_model_eval(element, (dfk, ifk))

    # Check that output is a valid kinematic chain
    check_valid_kinematic_chain_2d(dfk_out, ifk_out, dtype, device)


def check_kinematic_element_3d(
    element: nn.ModuleList,
    trainable: bool,
    dtype: torch.dtype,
    device: torch.device,
    allow_none_grad: bool = False,
) -> None:
    # Check that kinematic model can be evaluated and differentiated
    dfk, ifk = hom_identity_3d(dtype, device)

    if trainable:
        dfk_out, ifk_out = check_model_eval_and_grad(
            element, (dfk, ifk), allow_none_grad
        )
    else:
        dfk_out, ifk_out = check_model_eval(element, (dfk, ifk))

    # Check that output is a valid kinematic chain
    check_valid_kinematic_chain_3d(dfk_out, ifk_out, dtype, device)


def test_elements_2d() -> None:
    dtype, device = torch.float64, torch.device("cpu")
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)

    T = nn.Parameter(torch.tensor([5.0, 2.0]))

    elements_2d = nn.ModuleList(
        [
            Translate2D(),
            Translate2D(x=torch.tensor(0.1)),
            Translate2D(y=torch.tensor(0.2)),
            Translate2D(
                x=torch.tensor(0.1),
                y=torch.tensor(0.2),
            ),
            TranslateVec2D(T),
            Rotate2D(0.5),
            Rotate2D(torch.tensor(0.5)),
            AbsolutePosition2D(x=0.5),
            AbsolutePosition2D(y=-0.5),
            Gap(x=5.0),
            Gap(5.0),
        ]
    )

    for element in elements_2d:
        check_kinematic_element_2d(element, False, dtype, device)
        if not isinstance(element, AbsolutePosition2D):
            check_kinematic_element_2d(element.reverse(), False, dtype, device)


def test_trainable_elements_2d() -> None:
    dtype, device = torch.float64, torch.device("cpu")
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)

    T = nn.Parameter(torch.tensor([5.0, 2.0]))

    elements_2d = nn.ModuleList(
        [
            Translate2D(x=torch.tensor(0.1), trainable=True),
            Translate2D(y=torch.tensor(0.2), trainable=True),
            Translate2D(x=torch.tensor(0.1), y=torch.tensor(0.2), trainable=True),
            TranslateVec2D(T, trainable=True),
            Rotate2D(0.5, trainable=True),
            Rotate2D(torch.tensor(0.5), trainable=True),
            AbsolutePosition2D(x=0.5, trainable=True),
            AbsolutePosition2D(y=-0.5, trainable=True),
            Gap(x=5.0, trainable=True),
            Gap(5.0, trainable=True),
        ]
    )

    for element in elements_2d:
        print(list(element.named_parameters()))
        check_kinematic_element_2d(element, True, dtype, device)
        if not isinstance(element, AbsolutePosition2D):
            check_kinematic_element_2d(element.reverse(), True, dtype, device)


def test_elements_3d() -> None:
    dtype, device = torch.float64, torch.device("cpu")
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)

    T3d = torch.tensor([5.0, 2.0, -15.0])

    elements_3d = nn.ModuleList(
        [
            AbsolutePosition3D(
                torch.tensor(1.1),
                torch.tensor(1.2),
                torch.tensor(1.3),
            ),
            AbsolutePosition3D(
                x=torch.tensor(1.1),
            ),
            AbsolutePosition3D(
                y=torch.tensor(1.1),
            ),
            AbsolutePosition3D(
                z=torch.tensor(1.1),
            ),
            Translate3D(),
            Translate3D(x=torch.tensor(0.1)),
            Translate3D(y=torch.tensor(0.2)),
            Translate3D(z=torch.tensor(0.2)),
            Translate3D(
                x=torch.tensor(0.1),
                y=torch.tensor(0.2),
            ),
            Translate3D(
                x=torch.tensor(0.1),
                z=torch.tensor(0.2),
            ),
            Translate3D(
                y=torch.tensor(0.2),
                z=torch.tensor(0.2),
            ),
            Translate3D(
                x=torch.tensor(0.1),
                y=torch.tensor(0.2),
                z=torch.tensor(0.2),
            ),
            TranslateVec3D(T3d),
            Rotate3D(),
            Rotate3D(y=0.1),
            Rotate3D(z=0.2),
            Rotate3D(y=0.1, z=0.2),
            Rotate3D(y=torch.tensor(0.1)),
            Rotate3D(z=torch.tensor(0.2)),
            Rotate3D(
                y=torch.tensor(0.1),
                z=torch.tensor(0.2),
            ),
            Gap(x=5.0),
            Gap(5.0),
        ]
    )

    for element in elements_3d:
        check_kinematic_element_3d(element, False, dtype, device)
        if not isinstance(element, (AbsolutePosition3D, Rotate3D)):
            check_kinematic_element_3d(element.reverse(), False, dtype, device)


def test_trainable_elements_3d() -> None:
    dtype, device = torch.float64, torch.device("cpu")
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)

    T3d = torch.tensor([5.0, 2.0, -15.0])

    elements_3d = nn.ModuleList(
        [
            AbsolutePosition3D(
                torch.tensor(1.1),
                torch.tensor(1.2),
                torch.tensor(1.3),
                trainable=True,
            ),
            AbsolutePosition3D(
                x=torch.tensor(1.1),
                trainable=(True, False, False),
            ),
            AbsolutePosition3D(
                y=torch.tensor(1.1),
                trainable=(False, True, False),
            ),
            AbsolutePosition3D(
                z=torch.tensor(1.1),
                trainable=(False, False, True),
            ),
            Translate3D(
                x=torch.tensor(0.1),
                trainable=(True, False, False),
            ),
            Translate3D(
                y=torch.tensor(0.2),
                trainable=(False, True, False),
            ),
            Translate3D(
                z=torch.tensor(0.2),
                trainable=(False, False, True),
            ),
            Translate3D(
                x=torch.tensor(0.1),
                y=torch.tensor(0.2),
                trainable=True,
            ),
            Translate3D(
                x=torch.tensor(0.1), z=torch.tensor(0.2), trainable=(True, False, True)
            ),
            Translate3D(
                y=torch.tensor(0.2),
                z=torch.tensor(0.2),
                trainable=(False, True, True),
            ),
            Translate3D(
                x=torch.tensor(0.1),
                y=torch.tensor(0.2),
                z=torch.tensor(0.2),
                trainable=True,
            ),
            TranslateVec3D(T3d, trainable=True),
            Rotate3D(y=0.1, trainable=True),
            Rotate3D(z=0.2, trainable=True),
            Rotate3D(y=0.1, z=0.2, trainable=True),
            Rotate3D(y=torch.tensor(0.1), trainable=True),
            Rotate3D(z=torch.tensor(0.2), trainable=True),
            Rotate3D(y=torch.tensor(0.1), z=torch.tensor(0.2), trainable=True),
            Gap(x=5.0, trainable=True),
            Gap(5.0, trainable=True),
        ]
    )

    for element in elements_3d:
        check_kinematic_element_3d(element, True, dtype, device)
        if not isinstance(element, (AbsolutePosition3D, Rotate3D)):
            check_kinematic_element_3d(element.reverse(), True, dtype, device)


def test_elements_mixed() -> None:
    dtype, device = torch.float64, torch.device("cpu")
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)

    elements_mixed = nn.ModuleList(
        [
            Gap(5.0),
            Gap(torch.tensor(5.0)),
            Gap(x=5.0),
            Gap(x=torch.tensor(5.0)),
            Rotate((0.1, 0.2)),
            Translate(),
            Translate(x=0.1),
            Translate(y=0.2),
            Translate(z=0.3),
        ]
    )

    for element in elements_mixed:
        check_kinematic_element_2d(element, False, dtype, device)
        check_kinematic_element_3d(element, False, dtype, device)

        # TODO reverse mixed


def test_trainable_elements_mixed() -> None:
    dtype, device = torch.float64, torch.device("cpu")
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)

    elements_mixed = nn.ModuleList(
        [
            Gap(5.0, trainable=True),
            Gap(torch.tensor(5.0), trainable=True),
            Gap(x=5.0, trainable=True),
            Gap(x=torch.tensor(5.0), trainable=True),
            Rotate((0.1, 0.2), trainable=True),
            Translate(x=0.1, trainable=True),
            Translate(y=0.2, trainable=True),
            Translate(z=0.3, trainable=True),
        ]
    )

    for element in elements_mixed:
        check_kinematic_element_2d(element, True, dtype, device, allow_none_grad=True)
        check_kinematic_element_3d(element, True, dtype, device, allow_none_grad=True)

        # TODO reverse mixed


def test_elements_shared_parameter() -> None:
    """
    Test sharing parameters between elements
    """

    a = Gap(5.0, trainable=True)
    b = Gap(x=a.x)

    seq = KinematicSequential(a, b)

    assert len(list(seq.named_parameters())) == 1
    assert a.x is b.x