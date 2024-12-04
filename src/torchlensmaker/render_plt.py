import torch
import matplotlib.pyplot as plt
import numpy as np

from torchlensmaker.optics import OpticalSurface, FocalPointLoss


def draw_rays(ax, rays_origins, rays_ends):
    A = rays_origins.detach().numpy()
    B = rays_ends.detach().numpy()
    for a, b in zip(A, B):
        # draw rays
        ax.plot([a[0], b[0]], [a[1], b[1]], color="orange", linewidth=1.0, zorder=0)

        # draw rays origins
        ax.scatter(a[0], a[1], marker="x", color="lightgrey")
    
    print("rays aperture", np.max(A[:, 0]) - np.min(A[:, 0]))


def draw_surface_module(ax, surface, color):
    "Render surface to axes"

    # Render optical surface
    t = torch.linspace(*surface.domain(), 1000)
    points = surface.evaluate(t).detach().numpy()
    ax.plot(points[:, 0], points[:, 1], color=color)

    # Render surface anchor
    ax.plot(surface.pos[0].detach(), surface.pos[1].detach(), "+", color="grey")
    

def draw_surface_rays(ax, surface, inputs, outputs):
    if inputs is not None and outputs is not None:
        (rays_origins, _), _ = inputs[0]
        (rays_ends, _), _ = outputs
        draw_rays(ax, rays_origins, rays_ends)


def draw_focal_rays(ax, module, inputs, outputs):
    if inputs is not None and outputs is not None:
        (rays_origins, rays_vectors), _ = inputs[0]

        # Compute t needed to reach the focal point's position
        t_real = (module.pos[1] -rays_origins[:, 1])/rays_vectors[:, 1]

        if t_real.mean() > 0:
            t = 1.3 * t_real
        else:
            t = - t_real / 3
        
        end_x = rays_origins[:, 0] + t*rays_vectors[:, 0]
        end_y = rays_origins[:, 1] + t*rays_vectors[:, 1]
        draw_rays(ax, rays_origins, torch.column_stack((end_x, end_y)))
        

def render_element_module(ax, element):
    "Render optical element to axes"

    if isinstance(element, OpticalSurface):
        draw_surface_module(ax, element.surface, color="steelblue")
    
    elif isinstance(element, FocalPointLoss):
        pos = element.pos.detach().numpy()
        ax.plot(pos[0], pos[1], marker="+", markersize=5.0, color="red")


def render_element_rays(ax, module, inputs, outputs):
    # For surfaces, render input rays until the collision
    if isinstance(module, OpticalSurface):
        draw_surface_rays(ax, module, inputs, outputs)

    # For focal point loss, render rays up to a bit after the focal point
    elif isinstance(module, FocalPointLoss):
        draw_focal_rays(ax, module, inputs, outputs)

def render_all(ax, optics, num_rays):

    # To render, call forward() on the model
    # with a hook that catches input and outputs
    # and renders them at each step

    def forward_hook(module, inputs, outputs):
        render_element_module(ax, module)
        render_element_rays(ax, module, inputs, outputs)

    try:
        # Register hooks on all modules
        handles = []
        for module in optics.modules():
            handles.append(
                module.register_forward_hook(forward_hook)
            )

        # Call the forward model, this will call the hooks
        loss = optics(num_rays)

        # Remove all hooks
        for h in handles:
            h.remove()

    except RuntimeError as e:
        print("Error calling forward on model", e)



def render_plt(optics, inputs, force_uniform_source=True):
    
    fig, ax = plt.subplots(figsize=(12, 8))

    # TODO implement force_uniform_source
    
    render_all(ax, optics, inputs)


    plt.gca().set_title(f"")
    plt.gca().set_aspect("equal")

    plt.show()
