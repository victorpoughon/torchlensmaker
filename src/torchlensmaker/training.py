import math
import torch
import numpy as np
from os.path import join

import matplotlib.pyplot as plt

from .optics import (
    OpticalSurface,
    default_input,
    focal_point_loss,
)


def full_forward(module, inputs):
    """
    Evaluate a pytorch module, returning all intermediate inputs and outputs.
    
    `full_forward(module, inputs)` is equivalent to `module(inputs)`, except that
    instead of returning just the output, it also returns a list of
    (module, inputs, outputs) tuples where each tuple of the list corresponds to
    a single forward call in the module tree.

    The returned execution list does not include the top level forward call.

    Returns:
        execute_list: list of (module, inputs, outputs)
        outputs: final outputs of the top level module execution
    """

    execute_list = []

    # Define the forward hook
    def hook(mod, inp, out):
        execute_list.append((mod, inp, out))
    
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


def get_all_gradients(model):
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1))
    return torch.cat(grads)


def optimize(optics, optimizer, num_iter, nshow=20, regularization=None):
    viridis = plt.get_cmap('viridis')

    fig, ax = plt.subplots()

    # torch.autograd.detect_anomaly(True)

    parameters_record = {
        n: []
        for n, _ in optics.named_parameters()
    }

    loss_record = torch.zeros(num_iter)

    show_every = math.ceil(num_iter / nshow)
   
    for i in range(num_iter):

        optimizer.zero_grad()
    
        output = optics(default_input)

        # TODO for now we just assume last element is the focal point
        loss = focal_point_loss(output)

        if regularization is not None:
            loss = loss + regularization(optics)


        loss_record[i] = loss.detach()
        loss.backward()

        # Record parameter values
        for n, param in optics.named_parameters():
            parameters_record[n].append(param.detach().clone())

        grad = get_all_gradients(optics)
        if torch.isnan(grad).any():
            print("ERROR: nan in grad", grad)
            raise RuntimeError("nan in gradient, check your torch.where() =)")

        optimizer.step()
        
        if i % show_every == 0:
            iter_str = f"[{i:>3}/{num_iter}]"
            L_str = f"L= {loss.item():>6.3f} | grad norm= {torch.linalg.norm(grad)}"
            print(f"{iter_str} {L_str}")
            
            #for mod in optics.modules():   
                #if isinstance(mod, OpticalSurface) and mod.surface is not None:
                #    draw_surface_module(ax, mod.surface, color=viridis(i / num_iter))

    plt.gca().set_aspect("equal")
    plt.show()

    # Plot parameters and loss
    fig, (ax1, ax2) = plt.subplots(2, 1)
    epoch_range = np.arange(0, num_iter)
    ax2.plot(epoch_range, loss_record.detach())
    for n, param in optics.named_parameters():
        if parameters_record[n][0].dim() == 0:
            data = torch.stack(parameters_record[n]).detach().numpy()
            ax1.plot(epoch_range, data, label=n)
    ax1.set_title("parameter")
    ax1.legend()
    ax2.set_title("loss")
    plt.show()

