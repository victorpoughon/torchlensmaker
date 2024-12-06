import torch
import matplotlib.pyplot as plt
import numpy as np

from torchlensmaker.optics import OpticalSurface, FocalPointLoss, Aperture, default_input


def draw_rays(ax, rays_origins, rays_ends, color):
    A = rays_origins.detach().numpy()
    B = rays_ends.detach().numpy()
    for a, b in zip(A, B):
        # draw rays
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=1.0, zorder=0)

        # draw rays origins
        # ax.scatter(a[0], a[1], marker="x", color="lightgrey")
    
    if A.shape[0] > 0:
        print("rays aperture", np.max(A[:, 0]) - np.min(A[:, 0]))


def draw_surface_module(ax, surface, color):
    "Render surface to axes"

    # Render optical surface
    t = torch.linspace(*surface.domain(), 1000)
    points = surface.evaluate(t).detach().numpy()
    ax.plot(points[:, 0], points[:, 1], color=color)

    # Render surface anchor
    ax.plot(surface.pos[0].detach(), surface.pos[1].detach(), "+", color="grey")


def draw_aperture(ax, aperture, position, color):
    "Render aperture to axes"

    inner, outer = aperture.inner_width, aperture.outer_width

    X = np.array([inner/2, outer/2])
    Y = np.array([position[1], position[1]])
    ax.plot(X, Y, color=color)
    ax.plot(-X, Y, color=color)
    

def draw_surface_rays(ax, module, inputs, outputs):
    

    # If rays are blocks, render simply all rays from collision to collision
    if outputs.blocked is None:
        draw_rays(ax, inputs.rays_origins, outputs.rays_origins, color="orange")
    else:
        # Split into colliding and non colliding rays using blocked mask
        
        # Render non blocked rays
        blocked = outputs.blocked
        draw_rays(ax, inputs.rays_origins[~blocked], outputs.rays_origins, color="orange")

        # Render blocked rays up to the target
        rays_origins = inputs.rays_origins[blocked]
        rays_vectors = inputs.rays_vectors[blocked]
        if rays_origins.numel() > 0:
            pos = inputs.target
            t = (pos[1] - rays_origins[:, 1]) / rays_vectors[:, 1]

            end_x = rays_origins[:, 0] + t*rays_vectors[:, 0]
            end_y = rays_origins[:, 1] + t*rays_vectors[:, 1]
            draw_rays(ax, rays_origins, torch.column_stack((end_x, end_y)), color="lightgrey")



def draw_focal_rays(ax, module, inputs, outputs):
    rays_origins, rays_vectors = inputs.rays_origins, inputs.rays_vectors
    pos = inputs.target

    # Compute t needed to reach the focal point's position
    t_real = (pos[1] -rays_origins[:, 1])/rays_vectors[:, 1]

    if t_real.mean() > 0:
        t = 1.3 * t_real
    else:
        t = - t_real / 3
    
    end_x = rays_origins[:, 0] + t*rays_vectors[:, 0]
    end_y = rays_origins[:, 1] + t*rays_vectors[:, 1]
    draw_rays(ax, rays_origins, torch.column_stack((end_x, end_y)), color="orange")
        

def render_element_module(ax, element, inputs, outputs):
    "Render optical element to axes"

    if isinstance(element, OpticalSurface):
        draw_surface_module(ax, outputs.surface, color="steelblue")

    elif isinstance(element, Aperture):
        draw_aperture(ax, element, inputs.target.detach(), color="black")
    
    elif isinstance(element, FocalPointLoss):
        pos = inputs.target.detach().numpy()
        ax.plot(pos[0], pos[1], marker="+", markersize=5.0, color="red")


def render_element_rays(ax, module, inputs, outputs):
    # For surfaces, render input rays until the collision
    if isinstance(module, OpticalSurface):
        draw_surface_rays(ax, module, inputs, outputs)
    
    elif isinstance(module, Aperture):
        draw_surface_rays(ax, module, inputs, outputs)

    # For focal point loss, render rays up to a bit after the focal point
    elif isinstance(module, FocalPointLoss):
        draw_focal_rays(ax, module, inputs, outputs)

def render_all(ax, optics):

    # To render, call forward() on the model
    # with a hook that catches input and outputs
    # and renders them at each step

    def forward_hook(module, all_inputs, outputs):
        inputs, outputs = all_inputs[0], outputs

        render_element_module(ax, module, inputs, outputs)
        render_element_rays(ax, module, inputs, outputs)

    try:
        # Register hooks on all modules
        handles = []
        for module in optics.modules():
            handles.append(
                module.register_forward_hook(forward_hook)
            )

        # Call the forward model, this will call the hooks
        _ = optics(default_input)

        # Remove all hooks
        for h in handles:
            h.remove()

    except RuntimeError as e:
        print("Error calling forward on model", e)



def render_plt(optics, force_uniform_source=True):
    
    fig, ax = plt.subplots(figsize=(12, 8))

    # TODO implement force_uniform_source
    
    render_all(ax, optics)


    plt.gca().set_title(f"")
    plt.gca().set_aspect("equal")

    plt.show()
