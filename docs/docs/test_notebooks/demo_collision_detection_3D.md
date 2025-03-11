# Demo collision detection 3D


```python
import torchlensmaker as tlm
import torch
import torch.nn

from torchlensmaker.testing.basic_transform import basic_transform


dtype = torch.float64

def make_random_rays(num_rays, start_x, end_x, max_y, dtype):
    rays_start = (torch.rand((num_rays, 3), dtype=dtype) * 2 - 1) * max_y
    rays_start[:, 0] = start_x

    rays_end = (torch.rand((num_rays, 3)) * 2 - 1) * max_y
    rays_end[:, 0] = end_x

    rays_vectors = torch.nn.functional.normalize(rays_end - rays_start, dim=1)

    return torch.hstack((rays_start, rays_vectors))


test_rays = make_random_rays(
    num_rays=50,
    start_x=-15,
    end_x=200,
    max_y=50,
    dtype=dtype
)

test_data = [
    (basic_transform(1.0, "origin", [0., 10., 0.], [0., 0., 0.]), tlm.Sphere(15.0, 1e6)),
    (basic_transform(1.0, "origin", [0., 0., 0.], [10., 0., -10.]), tlm.Sphere(25.0, 20)),
    (basic_transform(1.0, "origin", [0., 0., 0.], [20., 20., 0.]), tlm.Sphere(15.0, -10)),
    (basic_transform(1.0, "origin", [0., 0., 0.], [30., 0., 0.]), tlm.Parabola(15., -0.05)),
    (basic_transform(1.0, "origin", [0., 0., 0.], [40., 0., 0.]), tlm.Parabola(20., -0.04)),
    (basic_transform(1.0, "origin", [0., 0., 0.], [50., 0., 0.]), tlm.Parabola(30., 0.02)),
    (basic_transform(1.0, "origin", [0., 10., -10.], [60., 0., 0.]), tlm.Parabola(30., 0.05)),
    (basic_transform(1.0, "origin", [0., 0., 0.], [80., 0., 0.]), tlm.CircularPlane(50.)),
    (basic_transform(1.0, "origin", [0., 0., 0.], [5., 0., -5.]), tlm.SquarePlane(15.)),
    
    (basic_transform(1.0, "origin", [0.0, 10.0, -10.0], [100.0, 0.0, 0.0]), tlm.Parabola(30.0, -0.05)),
    (basic_transform(-1.0, "extent", [0.0, 20.0, -20.0], [100.0, 2.0, 5.0]), tlm.Parabola(20.0, 0.05)),

    (basic_transform(1.0, "origin", [0.0, 10.0, -10.0], [100.0, 0.0, 0.0]), tlm.Parabola(30.0, 0.05)),
    (basic_transform(1.0, "extent", [0.0, 20.0, -20.0], [100.0, 2.0, 5.0]), tlm.Parabola(20.0, 0.05)),

    (basic_transform(1.0, "extent", [0.0,  0.0, 0.0], [50.0, 5.0, 5.0]), tlm.Parabola(30.0, 0.05)),
    (basic_transform(1.0, "extent", [0.0, 10.0, 0.0], [50.0, 5.0, 5.0]), tlm.Parabola(30.0, 0.05)),
    (basic_transform(1.0, "extent", [0.0, 20.0, 0.0], [50.0, 5.0, 5.0]), tlm.Parabola(30.0, 0.05)),
    (basic_transform(1.0, "extent", [0.0, 30.0, 0.0], [50.0, 5.0, 5.0]), tlm.Parabola(30.0, 0.05)),
    (basic_transform(1.0, "extent", [0.0, 40.0, 0.0], [50.0, 5.0, 5.0]), tlm.Parabola(30.0, 0.05)),

    (basic_transform(1.0, "origin", [0.0, 0.0, 0.0], [10.0, 0.0, 0.0]), tlm.Parabola(30., 0.05)),
    (basic_transform(1.0, "origin", [0.0, 0.0, 0.0], [10.0, 0.0, 0.0]), tlm.SquarePlane(30.)),

    (basic_transform(1.0, "extent", [0., 45., 0.], [150, 0., 0.]), tlm.Sphere(50., 50)),
    (basic_transform(-1.0, "extent", [0., 45., 0.], [150., 0., 0.]), tlm.Sphere(50., 50)),

    (basic_transform(1.0, "extent", [0., 0., 0.], [200, 0., 0.]), tlm.SphereR(50., 25)),
    (basic_transform(-1.0, "extent", [0., 0., 0.], [200., 0., 0.]), tlm.SphereR(50., 25)),

    (basic_transform(1.0, "origin", [45., 5., 5.], [-10., 10., -5.]), tlm.SquarePlane(15.)),

    (basic_transform(1.0, "extent", [0., 45., 0.], [150, 30., 0.]), tlm.Asphere(diameter=20, R=-15, K=-1.2, A4=0.00045)),
    (basic_transform(-1.0, "extent", [0., 45., 0.], [150., 30., 0.]), tlm.Asphere(diameter=20, R=-15, K=-1.2, A4=0.00045)),
]

test_surfaces = [s for t, s in test_data]
test_transforms = [t for t, s in test_data]


def demo(rays):

    all_points = torch.empty((0, 3))
    all_normals = torch.empty((0, 3))
    P, V = test_rays[:, :3], test_rays[:, 3:6]

    for transform, surface in test_data:

        points, normals, _ = tlm.intersect(surface, P, V, transform(surface))

        if points.numel() > 0:
            all_points = torch.cat((all_points, points), dim=0)
            all_normals = torch.cat((all_normals, normals), dim=0)

    rays_start = P
    rays_end = P + 350*V
    realized_transforms = [t(s) for t, s in zip(test_transforms, test_surfaces)]
    
    scene = tlm.viewer.new_scene("3D")
    scene["data"].append(tlm.viewer.render_rays(rays_start, rays_end, 0))
    scene["data"].extend(tlm.viewer.render_collisions(all_points, all_normals))
    scene["data"].append(tlm.viewer.render_surfaces(test_surfaces, realized_transforms, dim=3))
    
    tlm.viewer.display_scene(scene)


demo(test_rays)
```


<TLMViewer src="./demo_collision_detection_3D_tlmviewer/demo_collision_detection_3D_0.json" />

