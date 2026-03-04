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

from typing import Callable, Any, Self
from dataclasses import dataclass, field


@dataclass
class DeepForwardContextManager:
    model: nn.Module
    hooks: list[Any] = field(default_factory=list)
    inputs: dict[str, tuple[torch.Tensor, ...]] = field(default_factory=dict)
    outputs: dict[str, tuple[torch.Tensor, ...]] = field(default_factory=dict)

    def __enter__(self) -> Self:
        def make_hook(path: str) -> Any:
            def hook(mod: nn.Module, ins: Any, outs: Any) -> None:
                if isinstance(mod, MultiForwardModule):
                    # For multiforward inputs, skip the first two arguments
                    # which are there only to implement multiforward
                    self.inputs[path] = ins[2] if len(ins) == 3 else ins[2:]
                else:
                    # Remove the wrapping tuple if there is only one input
                    self.inputs[path] = ins[0] if len(ins) == 1 else ins
                self.outputs[path] = outs

            return hook

        for path, module in self.model.named_modules():
            self.hooks.append(module.register_forward_hook(make_hook("." + path)))
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        for h in self.hooks:
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
        
    Keys of the returned object are the full qualified module path names with a leading dot.
    Supports MultiForwardModule

    Args:
        model: PyTorch nn.Module to hook
    
    Returns:
        DeepForwardContextManager object that contains all inputs and outputs

    """
    return DeepForwardContextManager(model)


class MultiForwardModule(nn.Module):
    """
    Enable defining multiple forward functions with the @multiforward decorator
    and still have hooks called correctly
    """
    def forward(
        self, actual_forward: Callable[[Any], Any], *args: Any, **kwargs: Any
    ) -> Any:
        return actual_forward(*args, **kwargs)


def multiforward(new_forward_func: Callable[[Any], Any]) -> Callable[[Any], Any]:
    def wrapper(self: MultiForwardModule, *args: Any, **kwargs: Any) -> Any:
        return self(new_forward_func, self, *args, **kwargs)

    return wrapper
