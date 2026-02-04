# Test ray cylinder collision

## 3D: `rays_cylinder_collision`


```python
import torchlensmaker as tlm

import math
import torch

from torchlensmaker.core.cylinder_collision import rays_cylinder_collision

def uniform_disk_sampling(N, diameter, dtype):
    n = torch.tensor(math.ceil(math.sqrt(N)), dtype=torch.int64)
    return diameter * tlm.disk_sampling(n, n, dtype, torch.device("cpu"))

def make_random_rays(num_rays, start_x, end_x, max_y, dtype):
    rays_start = (torch.rand((num_rays, 3), dtype=dtype) * 2 - 1) * max_y
    rays_start[:, 0] = start_x

    rays_end = (torch.rand((num_rays, 3)) * 2 - 1) * max_y
    rays_end[:, 0] = end_x

    rays_vectors = torch.nn.functional.normalize(rays_end - rays_start, dim=1)

    return torch.hstack((rays_start, rays_vectors))


def horizontal_rays(num_rays, start_x, diameter):
    points = uniform_disk_sampling(num_rays, diameter, torch.float64)

    P = torch.cat((torch.full((num_rays, 1), start_x), points), dim=-1)
    V = torch.tile(torch.tensor([1.0, 0., 0.], dtype=torch.float64), (num_rays, 1))

    return P, V

def zerox_rays(num_rays, start_x, diameter):
    "Rays with Vx == 0"

    P = (torch.rand((num_rays, 3)) - 0.5) * diameter
    P[:, 0] = start_x
    V = (torch.rand((num_rays, 3)) - 0.5) * diameter
    V[:, 0] = 0.
    V = torch.nn.functional.normalize(V, dim=-1)

    return P, V

def edge_rays(num_rays, start_x, tau):
    theta = torch.linspace(0., 2 * torch.pi, num_rays)
    x = tau * torch.cos(theta)
    y = tau * torch.sin(theta)
    P = torch.stack([torch.full((num_rays,), start_x), x, y], dim=1)
    V = torch.tile(torch.tensor([1.0, 0., 0.], dtype=torch.float64), (num_rays, 1))

    return P, V

def demo():
    test_rays = make_random_rays(
        num_rays=50,
        start_x=-5,
        end_x=5,
        max_y=12,
        dtype=torch.float64
    )

    P, V = test_rays[:, :3], test_rays[:, 3:6]

    # add some perfectly horizontal rays
    #P, V = horizontal_rays(100, -5, 12)
    #P, V = edge_rays(100, -5, 10 / 2)
    #P, V = zerox_rays(100, 0, 10)

    surface = tlm.Sphere(10, R=-6)

    # Compute ray cylinder intersection
    xmin, xmax, tau = surface.bcyl()
    t1, t2, hit_mask = rays_cylinder_collision(P, V, xmin, xmax, tau)

    point1 = P + t1.unsqueeze(-1).expand_as(V)*V
    point2 = P + t2.unsqueeze(-1).expand_as(V)*V

    valid_point1 = point1[hit_mask]
    valid_point2 = point2[hit_mask]

    # Setup scene for rendering
    scene = tlm.new_scene("3D")

    # Render surface
    scene["data"].append(tlm.render_surface_local(surface, 3))

    # Render points
    scene["data"].append(tlm.render_points(valid_point1))
    scene["data"].append(tlm.render_points(valid_point2))
    
    # Render rays
    rays_length=30
    rays_start = P
    rays_end = P + rays_length*V
    scene["data"].append(
        tlm.render_rays(rays_start, rays_end, layer=0)
    )

    # Display
    scene["controls"] = {"show_bounding_cylinders": True}
    tlm.display_scene(scene)
    

demo()
```


<TLMViewer src="./test_cylinder_collision_files/test_cylinder_collision_0.json?url" />


## 2D: `ray_box_collision`


```python
import torchlensmaker as tlm
import torch

from torchlensmaker.core.cylinder_collision import rays_rectangle_collision
from torchlensmaker.testing.collision_datasets import NormalRays, FixedRays, OrbitalRays


def make_random_rays(num_rays, start_x, end_x, max_y, dtype):
    rays_start = (torch.rand((num_rays, 2), dtype=dtype) * 2 - 1) * max_y
    rays_start[:, 0] = start_x

    rays_end = (torch.rand((num_rays, 2)) * 2 - 1) * max_y
    rays_end[:, 0] = end_x

    rays_vectors = torch.nn.functional.normalize(rays_end - rays_start, dim=1)

    return torch.hstack((rays_start, rays_vectors))


def horizontal_rays(num_rays, start_x, diameter):
    points = uniform_disk_sampling(num_rays, diameter, torch.float64)

    P = torch.cat((torch.full((num_rays, 1), start_x), points), dim=-1)
    V = torch.tile(torch.tensor([1.0, 0., 0.], dtype=torch.float64), (num_rays, 1))

    return P, V

def zerox_rays(num_rays, start_x, diameter):
    "Rays with Vx == 0"

    P = (torch.rand((num_rays, 3)) - 0.5) * diameter
    P[:, 0] = start_x
    V = (torch.rand((num_rays, 3)) - 0.5) * diameter
    V[:, 0] = 0.
    V = torch.nn.functional.normalize(V, dim=-1)

    return P, V

def edge_rays(num_rays, start_x, tau):
    theta = torch.linspace(0., 2 * torch.pi, num_rays)
    x = tau * torch.cos(theta)
    y = tau * torch.sin(theta)
    P = torch.stack([torch.full((num_rays,), start_x), x, y], dim=1)
    V = torch.tile(torch.tensor([1.0, 0., 0.], dtype=torch.float64), (num_rays, 1))

    return P, V

def demo():
    test_rays = make_random_rays(
        num_rays=50,
        start_x=-5,
        end_x=5,
        max_y=12,
        dtype=torch.float64
    )

    #P, V = test_rays[:, :2], test_rays[:, 2:4]

    # add some perfectly horizontal rays
    #P, V = horizontal_rays(100, -5, 12)
    #P, V = edge_rays(100, -5, 10 / 2)
    #P, V = zerox_rays(100, 0, 10)

    #surface = tlm.Sphere(10, R=-6)
    surface = tlm.SagSurface(10, tlm.SagSum([
        tlm.Conical(C=torch.tensor(0.01999999955296516), K=torch.tensor(1.)),
        tlm.Aspheric(coefficients=torch.tensor([-0.004999999888241291]))
    ]))

    generator = OrbitalRays(dim=2, N=15, radius=1.1, offset=0.0, epsilon=0.0)
    P, V = generator(surface)

    # Compute ray cylinder intersection
    xmin, xmax, tau = surface.bcyl()

    t1, t2, hit_mask = rays_rectangle_collision(P, V, xmin, xmax, -tau, tau)

    point1 = P + t1.unsqueeze(-1).expand_as(V)*V
    point2 = P + t2.unsqueeze(-1).expand_as(V)*V

    valid_point1 = point1[hit_mask]
    valid_point2 = point2[hit_mask]

    # Setup scene for rendering
    scene = tlm.new_scene("2D")

    # Render surface
    scene["data"].append(tlm.render_surface_local(surface, 2))

    # Render points
    scene["data"].append(tlm.render_points(valid_point1))
    scene["data"].append(tlm.render_points(valid_point2))
    
    # Render rays
    rays_length=30
    rays_start = P
    rays_end = P + rays_length*V
    scene["data"].append(
        tlm.render_rays(rays_start, rays_end, layer=0)
    )

    # Display
    scene["controls"] = {"show_bounding_cylinders": True}
    tlm.display_scene(scene)
    

demo()
```


<TLMViewer src="./test_cylinder_collision_files/test_cylinder_collision_1.json?url" />

