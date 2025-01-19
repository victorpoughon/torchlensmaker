import torch.nn as nn

from typing import Any


def full_forward(
    module: nn.Module, inputs: Any
) -> tuple[list[tuple[nn.Module, Any, Any]], Any]:
    """
    Forward evaluate a model, but returns all intermediate inputs and outputs.

    This is kind of like normal forward evaluation of a model, as in:

        > outputs = model(inputs)

    except that all intermediate layers inputs and outputs are returned as a
    list of tree element tuples (module, inputs, outputs):

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
        execute_list.append((mod, inp[0], out))

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
