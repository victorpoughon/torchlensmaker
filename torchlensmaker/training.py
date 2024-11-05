import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from os.path import join

import matplotlib.pyplot as plt

from .optics import (
    FocalPointLoss,
    ParallelBeamUniform,
    ParallelBeamRandom,
    FixedGap,
    RefractiveSurface,
    OpticalStack,
    Lens,
    Anchor,
)


def render_rays(ax, rays_origins, rays_ends):
    A = rays_origins.detach().numpy()
    B = rays_ends.detach().numpy()
    for a, b in zip(A, B):
        # draw rays
        ax.plot([a[0], b[0]], [a[1], b[1]], color="orange", linewidth=1.0, zorder=0)

        # draw rays origins
        ax.scatter(a[0], a[1], marker="x", color="lightgrey")
    
    print("rays aperture", np.max(A[:, 0]) - np.min(A[:, 0]))

def render_module(ax, current_gap, module, inputs, outputs, surface_color):
    "If inputs or output or None, don't render rays"
    
    if isinstance(module, RefractiveSurface):
        # Render optical surface
        t = torch.linspace(*module.surface.domain(), 1000)
        points = module.surface.evaluate(t).detach().numpy()

        anchor_edge_offset = module.surface.evaluate(module.surface.domain()[1:])[0][1].detach().numpy()
        offset_in = anchor_edge_offset if module.anchors[0] == Anchor.Edge else 0.

        ax.plot(points[:, 0], - offset_in + current_gap + points[:, 1], color=surface_color)

        # Offset the rays end points because they are expressed in the next elements reference
        offset_out = anchor_edge_offset if module.anchors[1] == Anchor.Edge else 0.

        # Render input rays
        if inputs is not None and outputs is not None:
            rays_origins, _ = inputs
            rays_ends, _ = outputs
            A = (rays_origins + torch.tensor([0., current_gap]))
            B = (rays_ends + torch.tensor([0., current_gap - offset_in + offset_out]))
            render_rays(ax, A, B)

    
    if isinstance(module, FocalPointLoss):
        # Render focal point
        ax.plot(0., current_gap, marker="+", markersize=5.0, color="red")

        # Render input rays up to y=0
        if inputs is not None and outputs is not None:
            rays_origins, rays_vectors = inputs
            t = -rays_origins[:, 1]/rays_vectors[:, 1]
            end_x = rays_origins[:, 0] + t*rays_vectors[:, 0]
            A = (rays_origins + torch.tensor([0., current_gap]))
            B = torch.column_stack((end_x, torch.zeros_like(end_x) + current_gap))
            render_rays(ax, A, B)

def render(optics, num_rays, force_uniform_source=True):
    
    fig, ax = plt.subplots(figsize=(12, 8))

    # Integrate total gap to offset what we render
    total_gap = 0.

    def forward_hook(module, inputs, outputs):
        # Keep track of total gap offset along y axis
        nonlocal total_gap
        if isinstance(module, FixedGap):
            total_gap += module.origin
        
        # Render the module and associated rays
        render_module(ax, total_gap, module, inputs, outputs, surface_color="steelblue")

        if isinstance(module, RefractiveSurface):
            anchor_edge_offset = module.surface.evaluate(module.surface.domain()[1:])[0][1].detach().numpy()
            if module.anchors[0] == Anchor.Edge:
                total_gap -= anchor_edge_offset

            if module.anchors[1] == Anchor.Edge:
                total_gap += anchor_edge_offset

        # Force replace Random sources by uniform sources for rendering
        if force_uniform_source and isinstance(module, ParallelBeamRandom):
            return ParallelBeamUniform(module.radius).forward(inputs)
    
    # Forward model, using hook for rendering
    loss = optics.forward(num_rays, hook=forward_hook)
            
    
    plt.gca().set_title(f"")
    plt.gca().set_aspect("equal")
    plt.show()


def get_all_gradients(model):
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1))
    return torch.cat(grads)

def optimize(optics, optimizer, num_rays, num_iter, nshow=20, regularization=None):
    viridis = plt.get_cmap('viridis')

    fig, ax = plt.subplots()

    #torch.autograd.detect_anomaly(True)

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
            current_gap = 0.
            for mod in optics.modules():
                if isinstance(mod, FixedGap):
                    current_gap += mod.origin
                
                if isinstance(mod, RefractiveSurface):
                    render_module(ax, current_gap, mod, None, None, surface_color=viridis(i / num_iter))

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

