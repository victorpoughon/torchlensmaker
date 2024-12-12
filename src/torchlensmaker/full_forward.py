import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import Any

@dataclass
class ForwardContext:
    module: nn.Module
    inputs: Any
    outputs: Any

    def __iter__(self):
        return iter((self.module, self.inputs, self.outputs))


def full_forward(module, inputs):
    """
    Evaluate a pytorch module, returning all intermediate inputs and outputs.
    
    `full_forward(module, inputs)` is like `module(inputs)`, except that instead
    of returning just the output, it also returns a list of
    (module, inputs, outputs) tuples (actually ForwardContext objects) where
    each tuple of the list corresponds to a single forward call in the module
    tree.

    The returned list does not include the top level forward call.

    Returns:
        execute_list: list of (module, inputs, outputs)
        outputs: final outputs of the top level module execution
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
        outputs = module(inputs)
    finally:
        for h in hooks:
            h.remove()

    return execute_list, outputs
