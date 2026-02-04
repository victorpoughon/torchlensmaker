```python
import torchlensmaker as tlm
import torch
import math

from torchlensmaker.testing.collision_datasets import NormalRays, FixedRays, OrbitalRays
from torchlensmaker.testing.dataset_view import convergence_plot

from torchlensmaker.core.geometry import unit2d_rot, unit3d_rot, unit_vector


from typing import TypeAlias, Any

Tensor: TypeAlias = torch.Tensor

from torchlensmaker.core.collision_detection import CollisionAlgorithm, surface_f, default_collision_method
from torchlensmaker.core.cylinder_collision import rays_cylinder_collision, rays_rectangle_collision


def view_beams(surface, P, V, t, rays_length=50):

    N, D = P.shape
    B = t.shape[0]
    assert t.shape[1] == P.shape[0]
    
    # :: (B, N)

    # :: (B, N, D)
    points = P.expand((B, -1, -1)) + t.unsqueeze(-1).expand((B, N, D)) * V.expand((B, -1, -1))
    assert points.shape == (B, N, D)
    
    points = points.reshape((-1, D))
    assert points.shape == (B*N, D)
    
    scene = tlm.new_scene("2D" if D == 2 else "3D")
    scene["data"].append(tlm.render_surface_local(surface, D))
    
    rays_start = P - rays_length*V
    rays_end = P + rays_length*V
    scene["data"].append(
        tlm.render_rays(rays_start, rays_end, layer=0)
    )

    scene["data"].append(tlm.render_points(points))

    # Rays origins
    scene["data"].append(tlm.render_points(P, color="red"))
    

    tlm.display_scene(scene)


def view_coarse_phase(surface, P, V, results):
    B, N, HA = results.history_coarse.shape
    for h in range(HA):
        t = results.history_coarse[:, :, h]
        view_beams(surface, P, V, t)


###########

def demo():
    surface = tlm.SagSurface(10, tlm.SagSum([
        tlm.Conical(C=torch.tensor(0.01999999955296516), K=torch.tensor(1.)),
        tlm.Aspheric(coefficients=torch.tensor([-0.004999999888241291]))
    ]))
    
        
    dim = 2

    # this one fails with offset= -50 but not 50
    #generator = FixedRays(dim=2, N=35, direction=torch.tensor([0.1736, 0.9848], dtype=torch.float64), offset=-50.0, epsilon=0.05)

    #generator = FixedRays(dim=dim, N=12, direction=unit_vector(dim=dim), offset=10.0, epsilon=0.05)
    #generator = OrbitalRays(dim=2, N=15, radius=1.1, offset=0.0, epsilon=0.0)
    
    generator = FixedRays(dim=2, N=15, direction=unit2d_rot(10, dtype=torch.float64), offset=-30.0, epsilon=0.05)
    
    #generator = FixedRays(dim=3, N=15, direction=unit3d_rot(70., 5., dtype=torch.float64), offset=50.0, epsilon=0.05)

    P, V = generator(surface)


    # Collision detection instrumented

    N = P.shape[0]
    
    xmin, xmax, tau = surface.bcyl().unbind()

    if dim == 3:
        t1, t2, hit_mask = rays_cylinder_collision(P, V, xmin, xmax, tau)
    else:
        t1, t2, hit_mask = rays_rectangle_collision(P, V, xmin, xmax, -tau, tau)

    P_maybe, V_maybe = P[hit_mask], V[hit_mask]
    tmin, tmax = t1[hit_mask], t2[hit_mask]
    
    results = default_collision_method(surface, P_maybe, V_maybe, tmin, tmax, history=True)
    t = results.t
    local_points = P_maybe + t.unsqueeze(1).expand_as(V_maybe) * V_maybe

    #view_coarse_phase(surface, P, V, results)

    indices = hit_mask.nonzero().squeeze(-1)
    final_t = torch.zeros((N,), dtype=t.dtype).index_put(
        (indices,), t
    )
    
    view_beams(surface, P, V, final_t.unsqueeze(0))

    print("rmse:", surface.rmse(local_points))

    # TODO fix convergence plot
    #convergence_plot(surface, P, V, generator.__class__.__name__)

    assert torch.all(surface.contains(local_points) == True)

demo()
```


<TLMViewer src="./collision_detection_analysis_dataset_files/collision_detection_analysis_dataset_0.json?url" />


    rmse: 4.6940803599682113e-07

