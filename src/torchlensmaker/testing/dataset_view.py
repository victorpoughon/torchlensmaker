# This file is part of Torch Lens Maker
# Copyright (C) 2025 Victor Poughon
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch


import torchlensmaker as tlm

import matplotlib.pyplot as plt


def dataset_view(surface, P, V, rays_length=100):
    "View a collision dataset testcase with tlmviewer"

    dim = P.shape[-1]

    t, local_normals, valid = surface.local_collide(P, V)
    local_points = P + t.unsqueeze(-1).expand_as(V) * V

    scene = tlm.viewer.new_scene("2D" if dim == 2 else "3D")
    scene["data"].append(tlm.viewer.render_points(P, color="grey"))
    scene["data"].extend(tlm.viewer.render_collisions(local_points, local_normals))

    rays_start = P - rays_length * V
    rays_end = P + rays_length * V
    scene["data"].append(tlm.viewer.render_rays(rays_start, rays_end, layer=0))

    assert torch.all(torch.isfinite(P))
    assert torch.all(torch.isfinite(V))

    scene["data"].append(tlm.viewer.render_surface_local(surface, dim))
    tlm.viewer.display_scene(scene)
    # tlm.viewer.dump(scene, ndigits=2)


def convergence_plot(surface, P, V, dataset_name):
    "Plot convergence of iterative collision detection results"

    # Two axes for the coarse phase and the fine phase
    fig, (ax_coarse, ax_fine) = plt.subplots(2, 1, figsize=(10, 6), layout="tight")

    # Manually perform local_collide to get history
    results = surface.collision_method(surface, P, V, history=True)

    # Tensor shapes
    B, N, HA = results.history_coarse.shape
    _, HB = results.history_fine.shape
    D = P.shape[-1]

    ## COARSE PHASE PLOT

    # P and V go from (N, D) to (B, N, HA, D)
    P4 = P.unsqueeze(1).expand((B, N, HA, D))
    V4 = V.unsqueeze(1).expand((B, N, HA, D))
    points_history_coarse = (
        P4 + results.history_coarse.unsqueeze(-1).expand((B, N, HA, D)) * V4
    )
    assert points_history_coarse.shape == (B, N, HA, D)

    # For each ray, plot Q(t) of its best beam
    residuals_coarse = torch.ones((N, HA), dtype=surface.dtype)
    for ray_index in range(N):
        # F of each ray :: (B, HA)
        F = surface.Fd(points_history_coarse[:, ray_index, :, :])
        assert F.shape == (B, HA)

        _, bestF_indices = torch.min(torch.abs(F), dim=0)
        assert bestF_indices.shape == (HA,)
        Y = torch.gather(F, dim=0, index=bestF_indices.unsqueeze(0))
        assert Y.shape == (1, HA)
        ax_coarse.plot(
            range(HA),
            Y.squeeze(0),
            color="grey",
            label="Q(t)" if ray_index == 0 else None,
        )

        for h in range(HA):
            residuals_coarse[ray_index, h] = F[bestF_indices[h], h]

    # Total error computed over each rays best beam
    ax_coarse_error = ax_coarse.twinx()
    ax_coarse_error.set_yscale("log")
    ax_coarse_error.set_ylim([1e-10, 100])
    error_coarse = torch.sqrt(torch.sum(residuals_coarse**2, dim=0) / N)
    assert error_coarse.shape == (HA,)
    ax_coarse_error.plot(error_coarse, label="error (coarse phase)", color="coral")

    ## FINE PHASE PLOT

    # Reshape tensors for broadcasting
    P_expanded = P.unsqueeze(1)  # Shape: (N, 1, 2)
    V_expanded = V.unsqueeze(1)  # Shape: (N, 1, 2)
    t_history_expanded = results.history_fine.unsqueeze(2)  # Shape: (N, H, 1)

    # Compute points_history
    points_history = P_expanded + t_history_expanded * V_expanded  # Shape: (N, H, 2)

    assert results.history_fine.shape == (N, HB), (N, HB)
    assert points_history.shape == (N, HB, D)

    # plot Q(t) = F(P+tV) on
    for ray_index in range(N):
        ax_fine.plot(
            range(HB),
            surface.Fd(points_history[ray_index, :, :]),
            color="grey",
            label="Q(t)" if ray_index == 0 else None,
        )

    ax_fine.set_xlabel("iteration")
    ax_fine.set_ylabel("Q(t)")
    fig.suptitle(f"{dataset_name} | {surface.collision_method.name}")

    # plot total error
    ax_fine_error = ax_fine.twinx()
    ax_fine_error.set_ylabel("error")
    ax_fine_error.set_yscale("log")
    ax_fine_error.set_ylim([1e-10, 100])

    residuals = torch.ones((N, HB))
    for h in range(HB):
        residuals[:, h] = surface.Fd(points_history[:, h, :])

    error = torch.sqrt(torch.sum(residuals**2, dim=0) / N)
    assert error.shape == (HB,)
    ax_fine_error.plot(error, label="error (fine phase)", color="coral")

    # Combine labels into a single legend
    lines, labels = ax_fine.get_legend_handles_labels()
    lines2, labels2 = ax_fine_error.get_legend_handles_labels()
    ax_fine.legend(lines + lines2, labels + labels2, loc=0)

    return fig
