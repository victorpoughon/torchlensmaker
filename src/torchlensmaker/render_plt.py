import torch
import matplotlib.pyplot as plt
import numpy as np

from torchlensmaker.optics import RefractiveSurface, FocalPointLoss



def render_rays(ax, rays_origins, rays_ends):
    A = rays_origins.detach().numpy()
    B = rays_ends.detach().numpy()
    for a, b in zip(A, B):
        # draw rays
        ax.plot([a[0], b[0]], [a[1], b[1]], color="orange", linewidth=1.0, zorder=0)

        # draw rays origins
        ax.scatter(a[0], a[1], marker="x", color="lightgrey")
    
    print("rays aperture", np.max(A[:, 0]) - np.min(A[:, 0]))


def render_surface(ax, surface, color):
    "Render surface to axes"

    # Render optical surface
    t = torch.linspace(*surface.domain(), 1000)
    points = surface.evaluate(t).detach().numpy()
    ax.plot(points[:, 0], points[:, 1], color=color)

    # Render surface anchor
    ax.plot(surface.pos[0].detach(), surface.pos[1].detach(), "+", color="grey")
    

def render_element(ax, element):
    "Render optical element to axes"

    if isinstance(element, RefractiveSurface):
        render_surface(ax, element.surface, color="steelblue")
    
    elif isinstance(element, FocalPointLoss):
        ax.plot(element.pos[0], element.pos[1], marker="+", markersize=5.0, color="red")


def render_all(ax, optics, num_rays):

    # To render, call forward() on the model
    # with a hook that catches input and outputs
    # and renders them at each step

    def forward_hook(module, inputs, outputs):
        print("Forward hook on", module)
        # For surfaces, render rays collision to collision
        if isinstance(module, RefractiveSurface):
            render_element(ax, module)
            if inputs is not None and outputs is not None:
                print(inputs)
                (rays_origins, _), _ = inputs[0]
                (rays_ends, _), _ = outputs
                render_rays(ax, rays_origins, rays_ends)

        # For focal point loss, render rays up to a bit after the focal point
        elif isinstance(module, FocalPointLoss):
            render_element(ax, module)
            if inputs is not None and outputs is not None:
                (rays_origins, rays_vectors), _ = inputs[0]
                t = (module.pos[1] -rays_origins[:, 1])/rays_vectors[:, 1]
                t = 1.3*t
                end_x = rays_origins[:, 0] + t*rays_vectors[:, 0]
                end_y = rays_origins[:, 1] + t*rays_vectors[:, 1]
                render_rays(ax, rays_origins, torch.column_stack((end_x, end_y)))

    try:
        # Register hooks on all modules
        handles = []
        for module in optics.modules():
            print("registering on ", module)
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
