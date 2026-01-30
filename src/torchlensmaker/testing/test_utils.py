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


from typing import Any

import pytest
import torch
import torch.nn as nn


def check_model_eval(model: nn.Module, inputs: tuple[Any]) -> Any:
    "Evaluate a model forwards and run sanity checks"

    # Check the forward pass
    outputs: tuple[torch.Tensor, ...] | torch.Tensor = model(*inputs)
    if not isinstance(outputs, tuple):
        outputs = tuple(
            outputs,
        )
    assert [torch.isfinite(t).all() for t in outputs], (
        "Model outputs contain NaN or Inf"
    )

    return outputs


def check_model_eval_and_grad(
    model: nn.Module, inputs: tuple[Any], allow_none_grad: bool = False
) -> Any:
    """
    Evaluate a model forwards and backwards and run sanity checks
    Expects at least one trainable parameter
    """

    # Check the forward pass
    outputs: tuple[torch.Tensor, ...] | torch.Tensor = model(*inputs)
    if not isinstance(outputs, tuple):
        outputs = tuple(
            outputs,
        )
    assert [torch.isfinite(t).all() for t in outputs], (
        "Model outputs contain NaN or Inf"
    )

    # Check the backward pass
    parameters = list(model.named_parameters())
    assert len(parameters) > 0

    loss = torch.stack([t.sum() for t in outputs]).sum()
    model.zero_grad()
    loss.backward()  # type: ignore[no-untyped-call]
    for name, param in parameters:
        print(f"grad({name}) = {param.grad}")
        assert allow_none_grad or param.grad is not None
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), (
                f"Gradient of {name} contains NaN or Inf: {param.grad}"
            )

    return outputs
