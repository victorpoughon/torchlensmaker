#!/usr/bin/env python
# coding: utf-8

# Visualize implicit solver


from functools import partial

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import tlmviewer as tlmv
import torch
from jaxtyping import Float

import torchlensmaker as tlm

dtype, device = torch.float32, torch.device("cpu")

# sag = partial(tlm.parabolic_sag_2d, A=0.4)
# sag = partial(tlm.spherical_sag_2d, C=torch.tensor(1/1.5))

sag = partial(
    tlm.sag_sum_2d,
    sags=[
        partial(tlm.aspheric_sag_2d, params=torch.tensor([0.5, -0.05])),
        partial(tlm.conical_sag_2d, params=torch.tensor([-1 / 2.0, 0.5])),
    ],
)

# Init rays
rays_y = torch.linspace(-1.0, 1.0, 100, dtype=dtype)
rays_theta = torch.deg2rad(torch.linspace(-10.0, 10.0, 100))
P = torch.stack((torch.full_like(rays_y, -0.5), rays_y), dim=-1)
V = torch.stack((torch.cos(rays_theta), torch.sin(rays_theta)), dim=-1)


scene = tlm.new_scene("2D")

# render rays
start = P
end = P + 2.0 * V
scene.data.append(tlm.render_rays(start, end, 0, {}, {}))


def render_iter(num_iter, damping, color):
    # solve
    tau = torch.ones((), dtype=P.dtype, device=P.device)
    nf = torch.ones((), dtype=P.dtype, device=P.device)
    t = tlm.implicit_solver_newton(
        P,
        V,
        tlm.sag_to_implicit_2d_raw(sag, nf, tau),
        num_iter,
        damping,
        init="closest",
        clamp_positive=True,
    )

    # render collision points
    cp = P + t.unsqueeze(-1) * V
    node = tlm.render_points(cp, color, radius=0.005)

    return node


scene.data.append(render_iter(1, 1.0, "red"))
scene.data.append(render_iter(2, 1.0, "white"))
# scene.data.append(render_iter(3, 1.0, "blue"))
render_iter(3, 1.0, "white")

tlmv.push_scene(scene)
