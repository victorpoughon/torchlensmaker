# Demo collision detection 2D


```python
import torchlensmaker as tlm
import torch
import torch.nn
from pprint import pprint

from torchlensmaker.testing.basic_transform import basic_transform


def make_uniform_rays(num_rays, start_x, end_x, max_y, dim, dtype):
    start_x = torch.full((num_rays,), start_x, dtype=dtype)
    start_y = torch.linspace(0., max_y, num_rays, dtype=dtype)
    
    rays_start = torch.stack((start_x, start_y), dim=1)

    rays_end = rays_start.clone()
    rays_end[:, 0] = end_x

    rays_vectors = torch.nn.functional.normalize(rays_end - rays_start, dim=1)

    return torch.hstack((rays_start, rays_vectors))

def make_random_rays(num_rays, start_x, end_x, max_y, dim, dtype):
    rays_start = (torch.rand((num_rays, dim), dtype=dtype) * 2 - 1) * max_y
    rays_start[:, 0] = start_x

    rays_end = (torch.rand((num_rays, dim)) * 2 - 1) * max_y
    rays_end[:, 0] = end_x

    rays_vectors = torch.nn.functional.normalize(rays_end - rays_start, dim=1)

    return torch.hstack((rays_start, rays_vectors))

dtype = torch.float64

#test_rays = make_uniform_rays(num_rays=300, start_x=-10, end_x=40, max_y=13, dim=2, dtype=dtype)
test_rays = make_random_rays(num_rays=30, start_x=-50, end_x=50, max_y=15, dim=2, dtype=dtype)

Tensor = torch.Tensor


test_data = [
    (basic_transform(1.0, "extent", 0., [35/2., -5]), tlm.SphereR(35.0, 35/2)),
    (basic_transform(-1.0, "extent", 0., [35/2., -5]), tlm.SphereR(35.0, 35/2)),
    
    (basic_transform(1.0, "origin", 0., [0, 0]), tlm.SphereR(35.0, 35/2)),
    
    (basic_transform(1.0, "origin", 5., [15., -5]), tlm.Parabola(35.0, 0.010)),

    (basic_transform(1.0, "origin", -15., [25., 12]), tlm.Parabola(35.0, 0.010)),
    (basic_transform(-1.0, "origin", -15., [25., 12]), tlm.Parabola(35.0, 0.010)),

    (basic_transform(1.0, "extent", 15., [40., 12]), tlm.Parabola(35.0, 0.010)),
    (basic_transform(-1.0, "extent", 15., [40., 12]), tlm.Parabola(35.0, 0.010)),
    (basic_transform(1.0, "origin", 0., [40., 0]), tlm.Parabola(35.0, 0.010)),

    (basic_transform(1.0, "origin", 0., [-5., 0.]), tlm.SquarePlane(35.0)),
    (basic_transform(1.0, "origin", -40, [50., 0]), tlm.CircularPlane(35.0)),
    (basic_transform(1.0, "origin", -40, [51., 0]), tlm.CircularPlane(35.0)),

    (basic_transform(1.0, "origin", 0., [-7., 0.]), tlm.CircularPlane(20)),

    (basic_transform(-1.0, "extent", 0., [-15., 0.]), tlm.Sphere(diameter=15, R=18)),
    
    (basic_transform(-1.0, "origin", 0., [-30., 0.]), tlm.Asphere(diameter=20, R=-15, K=-1.2, A4=0.00045)),
    (basic_transform(-1.0, "origin", 0., [-35., 0.]), tlm.Asphere(diameter=20, R=-20, K=0.8, A4=-0.0003)),
    (basic_transform(-1.0, "origin", 0., [-40., 0.]), tlm.Asphere(diameter=20, R=-15, K=-1.2, A4=0.000045)),
    (basic_transform(-1.0, "origin", 0., [-45., 0.]), tlm.Asphere(diameter=20, R=1e6, K=0, A4=0.0)),
]

test_surfaces = [s for t, s in test_data]
test_transforms = [t for t, s in test_data]


def demo(rays):
    all_points = torch.empty((0, 2), dtype=dtype)
    all_normals = torch.empty((0, 2), dtype=dtype)
    P, V = test_rays[:, 0:2], test_rays[:, 2:4]

    # COLLIDE
    for transform, surface in test_data:
        points, normals, valid = tlm.intersect(surface, P, V, transform(surface))

        assert torch.sum(valid) == points.shape[0] == normals.shape[0]

        if points.numel() > 0:
            all_points = torch.cat((all_points, points), dim=0)
            all_normals = torch.cat((all_normals, normals), dim=0)

    # RENDER
    scene = tlm.viewer.new_scene("2D")
    scene["data"].extend(tlm.viewer.render_collisions(all_points, all_normals))

    rays_start = P
    rays_end = P + 100*V
    scene["data"].append(
        tlm.viewer.render_rays(rays_start, rays_end, 0)
    )

    scene["data"].append(tlm.viewer.render_surfaces(test_surfaces, [t(s) for t, s in test_data], dim=2))

     #pprint(scene)
    
    tlm.viewer.display_scene(scene)
    #tlm.viewer.ipython_display(scene, ndigits=3, dump=True)


demo(test_rays)
```


<TLMViewer src="./demo_collision_detection_2D_tlmviewer/demo_collision_detection_2D_0.json" />



```python
from torchlensmaker.testing.collision_datasets import make_samples3D

s = tlm.Sphere(30, 50)
samples2D = s.samples2D_half(100, epsilon=0.)

samples3D = make_samples3D(samples2D, M=4)

scene = tlm.viewer.new_scene("3D")
scene["data"].append(tlm.viewer.render_points(samples3D))
tlm.viewer.display_scene(scene)

```


<TLMViewer src="./demo_collision_detection_2D_tlmviewer/demo_collision_detection_2D_1.json" />

