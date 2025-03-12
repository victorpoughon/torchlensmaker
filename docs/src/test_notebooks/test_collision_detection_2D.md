# Demo collision detection 2D


```python
import torchlensmaker as tlm
import torch
import torch.nn
from pprint import pprint

from torchlensmaker.testing.basic_transform import basic_transform

from IPython.display import display, HTML

dtype = torch.float64

def make_uniform_rays(num_rays, start_x, end_x, max_y, dim, dtype):
    start_x = torch.full((num_rays,), start_x, dtype=dtype)
    start_y = torch.linspace(-max_y, max_y, num_rays, dtype=dtype)
    
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



#test_rays = make_uniform_rays(num_rays=30, start_x=-50, end_x=100, max_y=40, dim=2, dtype=dtype)
test_rays = make_random_rays(num_rays=50, start_x=-50, end_x=100, max_y=60, dim=2, dtype=dtype)

test_transforms = [
    basic_transform(1.0, "origin", 0, [100, 0]),
    basic_transform(1.0, "extent", 0, [100, 0]),
    
    basic_transform(-1.0, "origin", 0, [40., 0.]),
    basic_transform(-1.0, "extent", 0, [40., 0]),
    
    basic_transform(1.0, "origin", 40, [80., 0]),
    basic_transform(1.0, "origin", 90, [0., 0]),
    basic_transform(1.0, "extent", 90, [0., 0]),
]


def demo(title, surface):
    display(HTML(f"<h2>{title}</h2>"))
    all_points = torch.empty((0, 2), dtype=dtype)
    all_normals = torch.empty((0, 2), dtype=dtype)
    P, V = test_rays[:, 0:2], test_rays[:, 2:4]

    # COLLIDE
    for transform in test_transforms:
        points, normals, valid = tlm.intersect(surface, P, V, transform(surface))

        assert torch.sum(valid) == points.shape[0] == normals.shape[0]

        if points.numel() > 0:
            all_points = torch.cat((all_points, points), dim=0)
            all_normals = torch.cat((all_normals, normals), dim=0)

    # RENDER
    scene = tlm.viewer.new_scene("2D")
    scene["data"].extend(tlm.viewer.render_collisions(all_points, all_normals))

    rays_start = P
    rays_end = P + 250*V
    scene["data"].append(
        tlm.viewer.render_rays(rays_start, rays_end, layer=0)
    )

    scene["data"].append(tlm.viewer.render_surfaces([surface]*len(test_transforms), [t(surface) for t in test_transforms], dim=2))
    scene["title"] = title
    tlm.viewer.display_scene(scene)

demo("Sphere", tlm.Sphere(35.0, 36))
demo("SphereR", tlm.SphereR(35.0, 35/2))
demo("Parabola", tlm.Parabola(35.0, 0.05))
```


<h2>Sphere</h2>



<TLMViewer src="./test_collision_detection_2D_tlmviewer/test_collision_detection_2D_0.json" />



<h2>SphereR</h2>



<TLMViewer src="./test_collision_detection_2D_tlmviewer/test_collision_detection_2D_1.json" />



<h2>Parabola</h2>



<TLMViewer src="./test_collision_detection_2D_tlmviewer/test_collision_detection_2D_2.json" />

