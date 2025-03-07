import torch

from torchlensmaker.core.transforms import IdentityTransform
import torchlensmaker.viewer as viewer

import matplotlib.pyplot as plt


def dataset_view(surface, P, V, rays_length=100):
    "View a collision dataset testcase with tlmviewer"

    dim = P.shape[-1]

    t, local_normals, valid = surface.local_collide(P, V)
    local_points = P + t.unsqueeze(-1).expand_as(V) * V

    scene = viewer.new_scene("2D" if dim == 2 else "3D")
    scene["data"].append(viewer.render_points(P, color="grey"))
    scene["data"].extend(viewer.render_collisions(local_points, local_normals))
    

    rays_start = P - rays_length*V
    rays_end = P + rays_length*V
    scene["data"].append(
        viewer.render_rays(rays_start, rays_end, layer=0)
    )

    assert torch.all(torch.isfinite(P))
    assert torch.all(torch.isfinite(V))

    scene["data"].append(viewer.render_surface(surface))
    viewer.ipython_display(scene)
    #tlm.viewer.dump(scene, ndigits=2)


def convergence_plot(surface, P, V, dataset_name, methods):
    "Plot convergence of collision detection for multiple algorithms"


    # move rays by a tiny bit, to avoid t=0 local minimum
    # that happens with constructed datasets
    # TODO augment dataset with different shifts
    #P, V = move_rays(P, V, 0.)
    
    fig, axes = plt.subplots(len(methods), 2, figsize=(10, 3*len(methods)), layout="tight", squeeze=False)

    for i, method in enumerate(methods):
        axQ1, axQ2 = axes[i]

        t_solve, t_history = method(surface, P, V, history=True)
        
        # Reshape tensors for broadcasting
        N, H = P.shape[0], t_history.shape[1]
        P_expanded = P.unsqueeze(1)  # Shape: (N, 1, 2)
        V_expanded = V.unsqueeze(1)  # Shape: (N, 1, 2)
        t_history_expanded = t_history.unsqueeze(2)  # Shape: (N, H, 1)
    
        # Compute points_history
        points_history = P_expanded + t_history_expanded * V_expanded  # Shape: (N, H, 2)
    
        assert t_history.shape == (N, H), (N, H)
        assert points_history.shape == (N, H, 2)
    
        # plot Q(t)
        for ray_index in range(t_history.shape[0]):
            axQ2.plot(range(t_history.shape[1]), surface.f(points_history[ray_index, :, :]))
        
        #axQ1.set_xlabel("iteration")
        #axQ1.set_ylabel("Q(t)", rotation=0)
        axQ1.set_title(f"{dataset_name} | {str(method)}")

        # plot total error
        axE = axQ2.twinx()
        axE.set_ylabel("error")
        axE.set_yscale("log")
        axE.set_ylim([1e-8, 100])

        residuals = torch.ones((N, H))
        for h in range(H):
            residuals[:, h] = surface.f(points_history[:, h, :])

        error = torch.sqrt(torch.sum(residuals**2, dim=0) / N)
        assert error.shape == (H,)
        axE.plot(error, label="error")
        axE.legend()


    return fig
