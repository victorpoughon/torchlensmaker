import torch
import matplotlib.pyplot as plt
import numpy as np

import torchlensmaker as tlm

import matplotlib as mpl
viridis = mpl.colormaps['viridis']

def draw_rays(ax, rays_origins, rays_ends, color):
    A = rays_origins.detach().numpy()
    B = rays_ends.detach().numpy()
    for i, (a, b) in enumerate(zip(A, B)):

        # compute color if we have a color dimension
        if isinstance(color, str):
            this_color = color
        else:
            this_color = viridis(color[i])

        # draw rays
        ax.plot([a[0], b[0]], [a[1], b[1]], color=this_color, linewidth=1.0, zorder=0)

        # draw rays origins
        # ax.scatter(a[0], a[1], marker="x", color="lightgrey")

    if A.shape[0] > 0:
        print("rays aperture", np.max(A[:, 0]) - np.min(A[:, 0]))


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


        if color_dim == "base":
            color_data = outputs.coord_base
        elif color_dim == "object":
            color_data = outputs.coord_object
        else:
            color_data = "orange"

        rays_origins, rays_vectors = inputs.rays_origins, inputs.rays_vectors
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

        if color_dim == "base":
            color_data = outputs.coord_base
        elif color_dim == "object":
            color_data = outputs.coord_object
        else:
            color_data = "orange"

        # If rays are not blocked, render simply all rays from collision to collision
        if outputs.blocked is None:
            draw_rays(ax, inputs.rays_origins, outputs.rays_origins, color=color_data)
        else:
            # Split into colliding and non colliding rays using blocked mask

            # Render non blocked rays
            blocked = outputs.blocked
            draw_rays(
                ax, inputs.rays_origins[~blocked], outputs.rays_origins, color=color_data
            )

            # Render blocked rays up to the target
            rays_origins = inputs.rays_origins[blocked]
            rays_vectors = inputs.rays_vectors[blocked]
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
    tlm.FocalPointLoss: FocalPointArtist,
    #tlm.PointSource: PointSourceArtist,
}

default_sampling = {"base": 10, "object": 1}

def render_all(ax, optics, sampling, **kwargs):

    color_dim = kwargs.get("color_dim", None)

    execute_list, outputs = tlm.full_forward(optics, tlm.default_input, sampling)

    for module, inputs, outputs in execute_list:
        # Find matching artist and use it to render
        for typ, artist in artists_dict.items():
            if isinstance(module, typ):
                artist.draw_element(ax, module, inputs, outputs)
                if inputs.rays_origins.numel() > 0:
                    artist.draw_rays(ax, module, inputs, outputs, color_dim)
                break


def render_plt(optics, sampling=default_sampling, **kwargs):

    fig, ax = plt.subplots(figsize=(12, 8))

    render_all(ax, optics, sampling, **kwargs)

    plt.gca().set_title(f"")
    plt.gca().set_aspect("equal")

    plt.show()
