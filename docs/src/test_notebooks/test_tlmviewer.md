# Test tlmviewer


```python
# RENDERING RAYS

import torch
import torchlensmaker as tlm

start = torch.tensor([[0, 0], [0, 0]])
end = torch.tensor([[10, 10], [-10, -10]])
variables = {
    "base": torch.tensor([0.5, 0.6]),
    "object": torch.tensor([10.0, 11.0]),
}
domain = {
    "base": [0, 1],
    "object": [0, 12]
}

scene = tlm.viewer.new_scene("2D")
scene["data"].append(tlm.viewer.render_rays(start, end, 0, variables, domain))
tlm.viewer.display_scene(scene)
```


<TLMViewer src="./test_tlmviewer_tlmviewer/test_tlmviewer_0.json" />



```python
# RENDER 2D SURFACES

import torchlensmaker as tlm

from torchlensmaker.testing.basic_transform import basic_transform


test_data = [
    (basic_transform(1.0, "extent", 0., [35/2., 15]), tlm.SphereR(35.0, 35/2)),
    (basic_transform(-1.0, "extent", 0., [35/2., 15]), tlm.SphereR(35.0, 35/2)),

    (basic_transform(1.0, "extent", 50., [40., 5]), tlm.Parabola(125.0, 0.01)),

    (basic_transform(1.0, "extent", -50., [-40., 5]), tlm.CircularPlane(100.)),
    (basic_transform(1.0, "extent", 50., [-40., 5]), tlm.SquarePlane(100.)),
]

test_surfaces = [s for t, s in test_data]
test_transforms = [t(s) for t, s in test_data]

scene = tlm.viewer.new_scene("2D")


scene["data"].append(tlm.viewer.render_surfaces(test_surfaces, test_transforms, dim=2))

tlm.viewer.display_scene(scene)
#tlm.viewer.display_scene(scene, ndigits=3, dump=True)
```


<TLMViewer src="./test_tlmviewer_tlmviewer/test_tlmviewer_1.json" />



```python
# RENDER 3D SURFACES

import torchlensmaker as tlm

test_data = [
    (basic_transform(1.0, "origin", [45., 10., 0.], [0., 0., 0.]), tlm.Sphere(15.0, 1e6)),
    (basic_transform(1.0, "origin", [45., -10., 0.], [10., 0., -10.]), tlm.Sphere(25.0, 20)),
    (basic_transform(1.0, "origin", [45.0, -25.0, 0.0], [0.0, 0.0, 0.0]), tlm.Parabola(30., 0.05)),
    (basic_transform(1.0, "origin", [45.0, 0.0, 15.0], [-10.0, 10.0, 5.0]), tlm.SquarePlane(30.)),
    (basic_transform(1.0, "origin", [45., -60., 0.], [80., 0., 0.]), tlm.CircularPlane(50.)),
]

test_surfaces = [s for t, s in test_data]
test_transforms = [t(s) for t, s in test_data]

scene = tlm.viewer.new_scene("3D")
scene["data"].append(tlm.viewer.render_surfaces(test_surfaces, test_transforms, dim=3))

tlm.viewer.display_scene(scene)
#tlm.viewer.display_scene(scene,ndigits=2,dump=True)
```


<TLMViewer src="./test_tlmviewer_tlmviewer/test_tlmviewer_2.json" />

