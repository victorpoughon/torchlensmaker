import torch
import matplotlib.pyplot as plt
import numpy as np

import torchlensmaker as tlm

import matplotlib as mpl
viridis = mpl.colormaps['viridis']


def normalize_tensor(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor


def draw_rays(ax, rays_origins, rays_ends, color):
    A = rays_origins.detach().numpy()
    B = rays_ends.detach().numpy()

    if isinstance(color, torch.Tensor):
        color = normalize_tensor(color)

    for i, (a, b) in enumerate(zip(A, B)):

        # compute color if we have a color dimension
        if isinstance(color, str):
            this_color = color
        else:
            this_color = viridis(color[i].item())

        # draw rays
        ax.plot([a[0], b[0]], [a[1], b[1]], color=this_color, linewidth=1.0, zorder=0)

        # draw rays origins
        # ax.scatter(a[0], a[1], marker="x", color="lightgrey")


class Artist:
    @staticmethod
    def draw_element(ax, element, inputs, outputs):
        "Draw the optical element"
        raise NotImplementedError

    @staticmethod
    def draw_rays(ax, element, inputs, outputs):
        "Draw the input rays to the optical element"
        raise NotImplementedError


class FocalPointArtist(Artist):
    @staticmethod
    def draw_element(ax, element, inputs, outputs):
        pos = inputs.target.detach().numpy()
        ax.plot(pos[0], pos[1], marker="+", markersize=5.0, color="red")

    @staticmethod
    def draw_rays(ax, element, inputs, outputs, color_dim):
        if color_dim == "rays":
            color_data = outputs.rays.get("rays")
        elif color_dim == "object":
            color_data = outputs.rays.get("object")
        else:
            color_data = "orange"

        rays_origins, rays_vectors = inputs.rays.get(["RX", "RY"]), inputs.rays.get(["VX", "VY"])
        pos = inputs.target

        # Compute t needed to reach the focal point's position
        t_real = (pos[0] - rays_origins[:, 0]) / rays_vectors[:, 0]

        if t_real.mean() > 0:
            t = 1.3 * t_real
        else:
            t = -t_real / 3

        end_x = rays_origins[:, 0] + t * rays_vectors[:, 0]
        end_y = rays_origins[:, 1] + t * rays_vectors[:, 1]
        draw_rays(ax, rays_origins, torch.column_stack((end_x, end_y)), color=color_data)


class ImageArtist(Artist):
    @staticmethod
    def draw_element(ax, element, inputs, outputs):
        pos = inputs.target.detach().numpy()
        height = element.height.item()
        ax.plot([pos[0], pos[0]], [-height/2, height/2], linestyle="--", color="black")

    @staticmethod
    def draw_rays(ax, element, inputs, outputs, color_dim):
        if color_dim == "rays":
            color_data = outputs.rays.get("rays")
        elif color_dim == "object":
            color_data = outputs.rays.get("object")
        else:
            color_data = "orange"

        rays_origins, rays_vectors = (
            inputs.rays.get(["RX", "RY"]),
            inputs.rays.get(["VX", "VY"]),
        )
        pos = inputs.target

        # Compute t needed to reach the focal point's position
        t_real = (pos[0] - rays_origins[:, 0]) / rays_vectors[:, 0]

        if t_real.mean() > 0:
            t = 1.00 * t_real
        else:
            t = -t_real / 3

        end_x = rays_origins[:, 0] + t * rays_vectors[:, 0]
        end_y = rays_origins[:, 1] + t * rays_vectors[:, 1]
        draw_rays(ax, rays_origins, torch.column_stack((end_x, end_y)), color=color_data)


class PointSourceArtist(Artist):
    @staticmethod
    def draw_element(ax, element, inputs, outputs):
        pos = (inputs.target + torch.tensor([element.height, 0.])).detach().numpy()
        ax.plot(pos[0], pos[1], marker="o", fillstyle='none', markersize=2.0, color="orange")

    @staticmethod
    def draw_rays(ax, element, inputs, outputs):
        pass


class SurfaceArtist(Artist):
    @staticmethod
    def draw_element(ax, element, inputs, outputs):
        surface = element.surface(inputs.target)
        color = "steelblue"

        # Render optical surface
        t = torch.linspace(*surface.domain(), 1000)
        points = surface.evaluate(t).detach().numpy()
        ax.plot(points[:, 0], points[:, 1], color=color)

        # Render surface anchor
        ax.plot(surface.pos[0].detach(), surface.pos[1].detach(), "+", color="grey")

    @staticmethod
    def draw_rays(ax, element, inputs, outputs, color_dim):

        if color_dim == "rays":
            color_data = outputs.rays.get("rays").detach().numpy()
        elif color_dim == "object":
            color_data = outputs.rays.get("object").detach().numpy()
        else:
            color_data = "orange"

        # If rays are not blocked, render simply all rays from collision to collision
        if outputs.blocked is None:
            draw_rays(ax, inputs.rays_origins, outputs.rays_origins, color=color_data)
        else:
            # Split into colliding and non colliding rays using blocked mask

            # Render non blocked rays
            blocked = outputs.blocked
            input_origins = inputs.rays.get(["RX", "RY"])
            input_vectors = inputs.rays.get(["VX", "VY"])
            output_origins = outputs.rays.get(["RX", "RY"])

            draw_rays(
                ax, input_origins[~blocked], output_origins, color=color_data
            )

            # Render blocked rays up to the target
            rays_origins = input_origins[blocked]
            rays_vectors = input_vectors[blocked]
            if rays_origins.numel() > 0:
                pos = inputs.target
                t = (pos[0] - rays_origins[:, 0]) / rays_vectors[:, 0]

                end_x = rays_origins[:, 0] + t * rays_vectors[:, 0]
                end_y = rays_origins[:, 1] + t * rays_vectors[:, 1]
                draw_rays(
                    ax,
                    rays_origins,
                    torch.column_stack((end_x, end_y)),
                    color="lightgrey",
                )


class ApertureArtist(Artist):
    @staticmethod
    def draw_element(ax, element, inputs, outputs):
        position = inputs.target.detach().numpy()
        color = "black"

        X = np.array([position[0], position[0]])
        Y = np.array([element.diameter / 2, element.height / 2])
        ax.plot(X, Y, color=color)
        ax.plot(X, -Y, color=color)

    @staticmethod
    def draw_rays(ax, element, inputs, outputs, color_dim):
        SurfaceArtist.draw_rays(ax, element, inputs, outputs, color_dim)


artists_dict = {
    tlm.OpticalSurface: SurfaceArtist,
    tlm.Aperture: ApertureArtist,
    tlm.FocalPoint: FocalPointArtist,
    #tlm.PointSource: PointSourceArtist,
    tlm.Image: ImageArtist,
    tlm.ImagePlane: ImageArtist,
}

default_sampling = {"rays": 10, "object": 3}

def render_all(ax, optics, sampling, **kwargs):

    color_dim = kwargs.get("color_dim", None)

    execute_list, outputs = tlm.full_forward(optics, tlm.default_input, sampling)

    for module, inputs, outputs in execute_list:
        # Find matching artist and use it to render
        for typ, artist in artists_dict.items():
            if isinstance(module, typ):
                artist.draw_element(ax, module, inputs, outputs)
                if inputs.rays.numel() > 0:
                    artist.draw_rays(ax, module, inputs, outputs, color_dim)
                break


def render_plt(optics, sampling=default_sampling, **kwargs):

    fig, ax = plt.subplots(figsize=(12, 8))

    render_all(ax, optics, sampling, **kwargs)

    plt.gca().set_title(f"")
    plt.gca().set_aspect("equal")

    plt.show()
