import torch

from torchlensmaker.core.transforms import IdentityTransform
import torchlensmaker.viewer as viewer


def dataset_view(surface, dataset, rays_length=100):
    "View a collision dataset testcase with tlmviewer"

    P, V = dataset.P, dataset.V
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

    assert torch.all(torch.isfinite(dataset.P))
    assert torch.all(torch.isfinite(dataset.V))

    scene["data"].append(viewer.render_surfaces([surface], [IdentityTransform(dim=dim, dtype=surface.dtype)], dim=dim))
    scene["title"] = dataset.name
    viewer.ipython_display(scene)
    #tlm.viewer.dump(scene, ndigits=2)
