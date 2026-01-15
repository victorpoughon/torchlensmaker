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


from torchlensmaker.kinematics.homogeneous_geometry import (
    hom_identity_2d,
    hom_identity_3d,
    HomMatrix2D,
    HomMatrix3D,
)

from torchlensmaker.kinematics.kinematics_elements import (
    Translate2D,
    TranslateVec2D,
    Translate3D,
    TranslateVec3D,
    Rotate2D,
    Rotate3D,
    AbsolutePosition,
    AbsolutePositionVec3D,
    MixedDim,
    Gap,
    Rotate,
    Translate,
)


def check_model_eval_and_grad(model: nn.Module, inputs: tuple[Any]) -> Any:
    "Evaluate a model forwards and backwards and run sanity checks"

    # Check the forward pass
    outputs: tuple[torch.Tensor, ...] | torch.Tensor = model(*inputs)
    if not isinstance(outputs, tuple):
        outputs = tuple(
            outputs,
        )
    assert [torch.isfinite(t).all() for t in outputs], (
        "Model outputs contain NaN or Inf"
    )

    # Check the backward pass, if the model has any parameters
    parameters = list(model.named_parameters())
    if len(parameters) > 0:
        loss = torch.stack([t.sum() for t in outputs]).sum()
        model.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        for name, param in parameters:
            print(f"grad({name}) = {param.grad}")
            assert param.grad is not None
            assert torch.isfinite(param.grad).all(), (
                f"Gradient of {name} contains NaN or Inf: {param.grad}"
            )

    return outputs


def check_eval_kinematics_model_2d(
    elements: nn.ModuleList, dtype: torch.dtype, device: torch.device
) -> None:
    # Check that kinematic model can be evaluated and differentiated
    dfk, ifk = hom_identity_2d(dtype, device)

    for model in elements:
        dfk_out, ifk_out = check_model_eval_and_grad(model, (dfk, ifk))

        # Check that output is a valid kinematic chain
        check_valid_kinematic_chain_2d(dfk_out, ifk_out, dtype, device)


def check_eval_kinematics_model_3d(
    elements: nn.ModuleList, dtype: torch.dtype, device: torch.device
) -> None:
    # Check that kinematic model can be evaluated and differentiated
    dfk, ifk = hom_identity_3d(dtype, device)

    for model in elements:
        dfk_out, ifk_out = check_model_eval_and_grad(model, (dfk, ifk))

        # Check that output is a valid kinematic chain
        check_valid_kinematic_chain_3d(dfk_out, ifk_out, dtype, device)


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


def test_elements_2d() -> None:
    dtype, device = torch.float64, torch.device("cpu")
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)

    T = nn.Parameter(torch.tensor([5.0, 2.0]))

    chain_model_2d = nn.ModuleList(
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
        ]
    )

    check_eval_kinematics_model_2d(chain_model_2d, dtype, device)


def test_elements_3d() -> None:
    dtype, device = torch.float64, torch.device("cpu")
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)

    T3d = nn.Parameter(torch.tensor([5.0, 2.0, -15.0]))

    chain_model_3d = nn.ModuleList(
        [
            AbsolutePosition(
                torch.tensor(1.1),
                torch.tensor(1.2),
                torch.tensor(1.3),
            ),
            AbsolutePosition(
                x=torch.tensor(1.1),
            ),
            AbsolutePosition(
                y=torch.tensor(1.1),
            ),
            AbsolutePosition(
                z=torch.tensor(1.1),
            ),
            AbsolutePositionVec3D(
                torch.tensor([1.1, 1.2, 1.3]),
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
        ]
    )
    check_eval_kinematics_model_3d(chain_model_3d, dtype, device)


def test_elements_mixed() -> None:
    dtype, device = torch.float64, torch.device("cpu")
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)

    chain_model_mixed = nn.ModuleList(
        [
            MixedDim(
                Rotate2D(0.1),
                Rotate3D(y=0.1),
            ),
            Gap(5.0),
            Gap(torch.tensor(5.0)),
            # Rotate(),
            Rotate((0.1, 0.2)),
            Translate(),
            Translate(x=0.1),
            Translate(y=0.2),
            Translate(z=0.3),
        ]
    )

    check_eval_kinematics_model_2d(chain_model_mixed, dtype, device)
    check_eval_kinematics_model_3d(chain_model_mixed, dtype, device)
