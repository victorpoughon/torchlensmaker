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

import torch.nn as nn

from typing import Any, Iterator
from dataclasses import dataclass


@dataclass
class ModuleEvalContext:
    module: nn.Module
    inputs: Any
    outputs: Any

    def __iter__(self) -> Iterator[Any]:
        return iter((self.module, self.inputs, self.outputs))


def full_forward(module: nn.Module, inputs: Any) -> tuple[list[ModuleEvalContext], Any]:
    """
    Forward evaluate a model, but returns all intermediate inputs and outputs.

    This is kind of like normal forward evaluation of a model, as in:

        > outputs = model(inputs)

    except that all intermediate layers inputs and outputs are returned as a
    list of three element tuples (module, inputs, outputs):

        > execute_list, output = full_forward(model, inputs)
        > for module, inputs, outputs in execute_list:
        >     print(module, inputs, outputs)

    Args:
        module: PyTorch nn.Module to evaluate
        inputs: input data to the module

    Returns:
        execute_list: list of (module, inputs, outputs)
        outputs: output of the top level module execution
    """

    execute_list = []

    # Define the forward hook
    def hook(mod: nn.Module, inp: Any, out: Any) -> None:
        # inp[0] here restricts us to forward() first argument
        # so this only works with single argument forward() functions
        execute_list.append(ModuleEvalContext(mod, inp[0], out))

    # Register forward hooks to every module recursively
    hooks = []
    for mod in module.modules():
        hooks.append(mod.register_forward_hook(hook))

    # Evaluate the full model, then remove all hooks
    try:
        outputs = module(inputs)
    finally:
        for h in hooks:
            h.remove()

    return execute_list, outputs


def forward_tree(
    module: nn.Module, inputs: Any
) -> tuple[dict[nn.Module, Any], dict[nn.Module, Any]]:
    """
    Forward evaluate a model, but returns all intermediate inputs and outputs,
    in the form two dictionaries indexed by the modules.

    This is kind of like normal forward evaluation of a model, except that all
    intermediate layers inputs and outputs are returned as a two dictionaries:

        > input_tree, output_tree = tlm.forward_tree(inputs)
        > print(input_tree[module1])
        > print(output_tree[module2])

    Warning: Doesnt work if any modules are duplicated, because indexing of the
    returned dictionaries needs to be unique.

    Args:
        module: PyTorch nn.Module to evaluate
        inputs: input data to the module

    Returns:
        input_tree, output_tree: inputs and outputs dictionaries
    """

    input_tree: dict[nn.Module, Any] = {}
    output_tree: dict[nn.Module, Any] = {}

    # Define the forward hook
    def hook(mod: nn.Module, inp: Any, out: Any) -> None:
        # inp[0] here restricts us to forward() first argument
        # so this only works with single argument forward() functions
        input_tree[mod] = inp[0]
        output_tree[mod] = out

    # Register forward hooks to every module recursively
    hooks = []
    for mod in module.modules():
        hooks.append(mod.register_forward_hook(hook))

    # Evaluate the full model, then remove all hooks
    try:
        _ = module(inputs)
    finally:
        for h in hooks:
            h.remove()

    return input_tree, output_tree
