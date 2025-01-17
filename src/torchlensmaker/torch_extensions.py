import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import Any


# Aliases to torch.nn classes
Parameter = nn.Parameter


# Custom version of nn.Sequential that takes additional read only sampling info
class OpticalSequence(nn.Sequential):
    def forward(self, inputs, sampling):
        for module in self._modules.values():
            inputs = module(inputs, sampling)
        return inputs


@dataclass
class ForwardContext:
    module: nn.Module
    inputs: Any
    outputs: Any

    def __iter__(self):
        return iter((self.module, self.inputs, self.outputs))


def full_forward(module: nn.Module, inputs: Any, sampling: dict):
    """
    Evaluate an optical stack model

    This is kind of like normal forward evaluation of a model, as in: `outputs =
    model(inputs)`, except for two differences:

    1. All intermediate layers inputs and outputs are returned as a list of
       ForwardContext objects.
    2. The `sampling` object is passed to each module as additional read-only input.

    The returned list does not include the top level forward call.

    Returns:
        execute_list: list of (module, inputs, outputs) outputs: final outputs
        of the top level module execution
    """

    execute_list = []

    # Define the forward hook
    def hook(mod, inp, out):
        # inp[0] here restricts us to forward() first argument
        # this is fine here because we only need OpticalData
        execute_list.append(ForwardContext(mod, inp[0], out))
    
    # Register forward hooks to every module recursively
    hooks = []
    for mod in module.modules():
        hooks.append(mod.register_forward_hook(hook))

    # Evaluate the full model, then remove all hooks
    try:
        outputs = module(inputs, sampling)
    finally:
        for h in hooks:
            h.remove()

    return execute_list, outputs
