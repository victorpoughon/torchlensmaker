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

import torch
import torch.nn as nn

from typing import Any, Self, Callable
from dataclasses import dataclass, field

from torchlensmaker.core.base_module import MultiForwardModule


@dataclass
class DeepForwardContextManager:
    model: nn.Module
    _hooks: list[Any] = field(default_factory=list)
    inputs: dict[nn.Module, tuple[torch.Tensor, ...]] = field(default_factory=dict)
    outputs: dict[nn.Module, tuple[torch.Tensor, ...]] = field(default_factory=dict)

    # In the multiforward case, keep track of already called actual_forward
    # functions. This allow multiforward implementations to delegate to each
    # other without counting as multiple forward calls.
    _nexus: set[tuple[nn.Module, Callable[[Any], Any]]] = field(default_factory=set)

    def __enter__(self) -> Self:
        def hook_normal(mod: nn.Module, ins: Any, outs: Any) -> None:
            if mod in self.inputs or mod in self.outputs:
                raise RuntimeError(
                    f"deep_forward() error: model contains a duplicated module ({mod}) or multiple calls to the same module forward()"
                )
            # Remove the wrapping tuple if there is only one input
            self.inputs[mod] = ins[0] if len(ins) == 1 else ins
            self.outputs[mod] = outs

        def hook_multi(mod: nn.Module, ins: Any, outs: Any) -> None:
            key = (mod, ins[0])
            if key in self._nexus:
                raise RuntimeError(
                    f"deep_forward() error: (multiforward) model contains a duplicated module ({mod}) or multiple calls to the same module forward()"
                )

            self._nexus.add(key)

            if mod in self.inputs:
                return

            # For multiforward inputs, skip the first two arguments
            # which are there only to implement multiforward
            self.inputs[mod] = ins[2] if len(ins) == 3 else ins[2:]
            self.outputs[mod] = outs

        for _, module in self.model.named_modules():
            if isinstance(module, MultiForwardModule):
                self._hooks.append(module.register_forward_hook(hook_multi))
            else:
                self._hooks.append(module.register_forward_hook(hook_normal))
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        for h in self._hooks:
            h.remove()


def deep_forward(model: nn.Module) -> DeepForwardContextManager:
    """
    Forward evaluate a model, but returns store all intermediate layers inputs
    and outputs.

    Implemented as a context manager, so you can use it like:

        > with deep_forward(model) as trace:
        >     output = model(data)
        > trace.inputs[...]
        > trace.outputs[...]

    Keys of the returned object are the modules
    Supports MultiForwardModule

    Args:
        model: PyTorch nn.Module to hook

    Returns:
        DeepForwardContextManager object that contains all inputs and outputs

    """
    return DeepForwardContextManager(model)
