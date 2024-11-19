import math
import torch
import numpy as np
from os.path import join

import matplotlib.pyplot as plt

from .optics import (
    RefractiveSurface,
)

from torchlensmaker.render_plt import render_surface


def get_all_gradients(model):
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1))
    return torch.cat(grads)


def optimize(optics, optimizer, num_rays, num_iter, nshow=20, regularization=None):
    viridis = plt.get_cmap('viridis')

    fig, ax = plt.subplots()

    # torch.autograd.detect_anomaly(True)

    parameters_record = [
        torch.zeros((num_iter, p.shape[0]))
        for p in optics.parameters()
    ]

    loss_record = torch.zeros(num_iter)

    show_every = math.ceil(num_iter / nshow)
   
    for i in range(num_iter):
    
        optimizer.zero_grad()
        loss = optics.forward(num_rays)# + optics.lens.regularization()  # TODO refactor this into custom sequence

        if regularization is not None:
            loss += regularization(optics)

        loss_record[i] = loss
        loss.backward()

        # Record parameter values
        for j, param in enumerate(optics.parameters()):
            parameters_record[j][i, :] = param.detach()

        grad = get_all_gradients(optics)
        if torch.isnan(grad).any():
            print("ERROR: nan in grad", grad)
            raise RuntimeError("nan in gradient, check your torch.where() =)")
        
        optimizer.step()
        
        if i % show_every == 0:
            iter_str = f"[{i:>3}/{num_iter}]"
            L_str = f"L= {loss.item():>6.3f} | grad norm= {torch.linalg.norm(grad)}"
            print(f"{iter_str} {L_str}")
            for mod in optics.modules():
                
                if isinstance(mod, RefractiveSurface):
                    render_surface(ax, mod.surface, color=viridis(i / num_iter))

    plt.gca().set_aspect("equal")
    plt.show()

    # Plot parameters and loss
    fig, (ax1, ax2) = plt.subplots(2, 1)
    epoch_range = np.arange(0, num_iter)
    ax2.plot(epoch_range, loss_record.detach())
    for j, param in enumerate(optics.parameters()):
        ax1.plot(epoch_range, parameters_record[j].detach(), label=str(j))
    ax1.set_title("parameter")
    ax2.set_title("loss")
    plt.show()

