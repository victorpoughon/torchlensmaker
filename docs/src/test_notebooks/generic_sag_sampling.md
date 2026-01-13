```python
import torchlensmaker as tlm

sphere1 = tlm.Sphere(35, 50)
sphere2 = tlm.Sphere(10, 5.01)
sphereR = tlm.SphereR(35.0, -35/2)
parabola = tlm.Parabola(35.0, -0.010)
square_plane = tlm.SquarePlane(35.0)
circular_plane = tlm.CircularPlane(35.0)
asphere = tlm.Asphere(diameter=20, R=-15, K=-1.2, coefficients=[0.00045])

g = 10

optics = tlm.Sequential(
    tlm.ReflectiveSurface(sphere2, anchors=("extent", "extent")),
    tlm.Gap(g),
    tlm.ReflectiveSurface(sphereR),
    tlm.Gap(g),
    tlm.ReflectiveSurface(parabola),
    tlm.Gap(g),
    tlm.ReflectiveSurface(square_plane),
    tlm.Gap(g),
    tlm.ReflectiveSurface(circular_plane),
    tlm.Gap(g),
    tlm.ReflectiveSurface(asphere),
    tlm.Gap(g),
    tlm.ReflectiveSurface(sphere1)
)

tlm.show2d(optics, controls={"show_optical_axis": True, "show_other_axes": True, "show_kinematic_joints": True})

tlm.show3d(optics, controls={"show_optical_axis": True, "show_other_axes": True, "show_kinematic_joints": True})
```


<TLMViewer src="./generic_sag_sampling_files/generic_sag_sampling_0.json?url" />



<TLMViewer src="./generic_sag_sampling_files/generic_sag_sampling_1.json?url" />



```python
from pprint import pprint
import json
import torch

scene = tlm.render_sequence(optics, dim=3, dtype=torch.float64, sampling={})
print(json.dumps(scene))
```

    {"mode": "3D", "camera": "orthographic", "data": [{"type": "surface-sag", "diameter": 10, "sag-function": {"sag-type": "spherical", "C": 0.1996007984031936}, "bcyl": [0.0, 4.693614159608872, 5.0], "matrix": [[1.0, 0.0, 0.0, -4.693614159608872], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]}, {"type": "surface-sphere-r", "diameter": 35.0, "R": -17.5, "matrix": [[1.0, 0.0, 0.0, 10.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]}, {"type": "surface-sag", "diameter": 35.0, "sag-function": {"sag-type": "parabolic", "A": -0.01}, "bcyl": [-3.0625, 0.0, 17.5], "matrix": [[1.0, 0.0, 0.0, 20.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]}, {"type": "surface-plane", "radius": 24.748737341529164, "clip_planes": [[0.0, -1.0, 0.0, 17.5], [0.0, 1.0, 0.0, 17.5], [0.0, 0.0, -1.0, 17.5], [0.0, 0.0, 1.0, 17.5]], "matrix": [[1.0, 0.0, 0.0, 30.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]}, {"type": "surface-plane", "radius": 17.5, "clip_planes": [], "matrix": [[1.0, 0.0, 0.0, 40.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]}, {"type": "surface-sag", "diameter": 20, "sag-function": {"sag-type": "sum", "terms": [{"sag-type": "conical", "K": -1.2, "C": -0.06666666666666667}, {"sag-type": "aspheric", "coefficients": [0.00045]}]}, "bcyl": [-3.2623792124926396, 4.5, 10.0], "matrix": [[1.0, 0.0, 0.0, 50.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]}, {"type": "surface-sag", "diameter": 35, "sag-function": {"sag-type": "spherical", "C": 0.02}, "bcyl": [0.0, 3.1625150120120136, 17.5], "matrix": [[1.0, 0.0, 0.0, 60.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]}, {"type": "points", "data": [[0.0, 0.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[0.0, 0.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[10.0, 0.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[10.0, 0.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[20.0, 0.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[20.0, 0.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[30.0, 0.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[30.0, 0.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[40.0, 0.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[40.0, 0.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[50.0, 0.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[50.0, 0.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[60.0, 0.0, 0.0]], "layers": [4]}]}



```python
import torchlensmaker as tlm
import torch
import json

surface = tlm.Parabola(10, -0.02)

optics = tlm.Sequential(
    tlm.PointSourceAtInfinity(8),
    tlm.Gap(10),
    tlm.MixedDim(tlm.Rotate2D(5.0), tlm.Rotate3D(y=5.0, z=10.0)),
    tlm.ReflectiveSurface(surface)
)

tlm.show2d(optics, end=5)
tlm.show3d(optics, end=5)
```


<TLMViewer src="./generic_sag_sampling_files/generic_sag_sampling_2.json?url" />



<TLMViewer src="./generic_sag_sampling_files/generic_sag_sampling_3.json?url" />



```python
import torchlensmaker as tlm
import torch


optics = tlm.Sequential(
    tlm.ReflectiveSurface(tlm.Sphere(1., 1.55)),
    tlm.ReflectiveSurface(tlm.SagSurface(1., tlm.Conical(torch.tensor(1./0.8), torch.tensor(-100.05)))),
    tlm.Gap(1),
    tlm.ReflectiveSurface(tlm.SagSurface(1., tlm.Aspheric(torch.tensor([1e-0, 0., 0.])))),
    tlm.Gap(1),
    tlm.ReflectiveSurface(tlm.SagSurface(1., tlm.Aspheric(torch.tensor([0., 1e-0, 0.])))),
    tlm.Gap(1),
    tlm.ReflectiveSurface(tlm.SagSurface(1., tlm.Aspheric(torch.tensor([0., 0., 1e-0])))),
    tlm.Gap(1),
    tlm.ReflectiveSurface(tlm.SagSurface(1., tlm.Aspheric(torch.tensor([5e0, -2e1, 0.])))),
    tlm.Gap(1),
)

#tlm.show2d(optics)
tlm.show3d(optics)


from pprint import pprint
import json
import torch

scene = tlm.render_sequence(optics, dim=3, dtype=torch.float64, sampling={})
print(json.dumps(scene))
```


<TLMViewer src="./generic_sag_sampling_files/generic_sag_sampling_4.json?url" />


    {"mode": "3D", "camera": "orthographic", "data": [{"type": "surface-sag", "diameter": 1.0, "sag-function": {"sag-type": "spherical", "C": 0.6451612903225806}, "bcyl": [0.0, 0.08285992488787901, 0.5], "matrix": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]}, {"type": "surface-sag", "diameter": 1.0, "sag-function": {"sag-type": "conical", "K": -100.05000305175781, "C": 1.25}, "bcyl": [0.0, 0.04280756415923484, 0.5], "matrix": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]}, {"type": "surface-sag", "diameter": 1.0, "sag-function": {"sag-type": "aspheric", "coefficients": [1.0, 0.0, 0.0]}, "bcyl": [0.0, 0.0625, 0.5], "matrix": [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]}, {"type": "surface-sag", "diameter": 1.0, "sag-function": {"sag-type": "aspheric", "coefficients": [0.0, 1.0, 0.0]}, "bcyl": [0.0, 0.015625, 0.5], "matrix": [[1.0, 0.0, 0.0, 2.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]}, {"type": "surface-sag", "diameter": 1.0, "sag-function": {"sag-type": "aspheric", "coefficients": [0.0, 0.0, 1.0]}, "bcyl": [0.0, 0.00390625, 0.5], "matrix": [[1.0, 0.0, 0.0, 3.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]}, {"type": "surface-sag", "diameter": 1.0, "sag-function": {"sag-type": "aspheric", "coefficients": [5.0, -20.0, 0.0]}, "bcyl": [-0.3125, 0.3125, 0.5], "matrix": [[1.0, 0.0, 0.0, 4.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]}, {"type": "points", "data": [[0.0, 0.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[0.0, 0.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[0.0, 0.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[1.0, 0.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[1.0, 0.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[2.0, 0.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[2.0, 0.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[3.0, 0.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[3.0, 0.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[4.0, 0.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[4.0, 0.0, 0.0]], "layers": [4]}]}



```python
import torchlensmaker as tlm
import torch
import json

s1 = tlm.Spherical(tlm.parameter(torch.tensor(1.)))
s2 = tlm.Aspheric(tlm.parameter(torch.tensor([0.5, 1.0])))

s3 = tlm.SagSum([s1, s2])

surface = tlm.SagSurface(1., s1)

optics = tlm.Sequential(
    tlm.ReflectiveSurface(surface)
)

tlm.show3d(optics)

scene = tlm.render_sequence(optics, dim=3, dtype=torch.float64, sampling={})
print(json.dumps(scene))

```


<TLMViewer src="./generic_sag_sampling_files/generic_sag_sampling_5.json?url" />


    {"mode": "3D", "camera": "orthographic", "data": [{"type": "surface-sag", "diameter": 1.0, "sag-function": {"sag-type": "spherical", "C": 1.0}, "bcyl": [0.0, 0.13397459621556135, 0.5], "matrix": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]}, {"type": "points", "data": [[0.0, 0.0, 0.0]], "layers": [4]}]}



```python
import torchlensmaker as tlm
import torch
import json

asphere = tlm.SagSurface(30, tlm.SagSum([
    #tlm.Conical(C=torch.tensor(-1/15), K=torch.tensor(-1.6)),
    tlm.Parabolic(A=torch.tensor(-0.025)),
    tlm.Aspheric(torch.tensor([0.00012]))]))

optics = tlm.Sequential(
    tlm.ReflectiveSurface(tlm.Sphere(1., 0.6)),
    tlm.SubChain(
        tlm.MixedDim(tlm.Translate2D(), tlm.Translate3D(z=1)),
        tlm.ReflectiveSurface(tlm.SphereR(1., 0.6)),
    ),
    tlm.Gap(3),
    tlm.ReflectiveSurface(tlm.Sphere(1., -0.6)),
    tlm.SubChain(
        tlm.MixedDim(tlm.Translate2D(), tlm.Translate3D(z=1)),
        tlm.ReflectiveSurface(tlm.SphereR(1., -0.6)),
    ),

    tlm.Gap(5),
    tlm.ReflectiveSurface(asphere)
)

tlm.show2d(optics)
tlm.show3d(optics)

scene = tlm.render_sequence(optics, dim=3, dtype=torch.float64, sampling={})
print(json.dumps(scene))
```


<TLMViewer src="./generic_sag_sampling_files/generic_sag_sampling_6.json?url" />



<TLMViewer src="./generic_sag_sampling_files/generic_sag_sampling_7.json?url" />


    {"mode": "3D", "camera": "orthographic", "data": [{"type": "surface-sag", "diameter": 1.0, "sag-function": {"sag-type": "spherical", "C": 1.6666666666666667}, "bcyl": [0.0, 0.26833752096446006, 0.5], "matrix": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]}, {"type": "surface-sphere-r", "diameter": 1.0, "R": 0.6, "matrix": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]]}, {"type": "surface-sag", "diameter": 1.0, "sag-function": {"sag-type": "spherical", "C": -1.6666666666666667}, "bcyl": [-0.26833752096446006, 0.0, 0.5], "matrix": [[1.0, 0.0, 0.0, 3.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]}, {"type": "surface-sphere-r", "diameter": 1.0, "R": -0.6, "matrix": [[1.0, 0.0, 0.0, 3.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]]}, {"type": "surface-sag", "diameter": 30, "sag-function": {"sag-type": "sum", "terms": [{"sag-type": "parabolic", "A": -0.02500000037252903}, {"sag-type": "aspheric", "coefficients": [0.00011999999696854502]}]}, "bcyl": [-5.625000083819032, 6.074999846532592, 15.0], "matrix": [[1.0, 0.0, 0.0, 8.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]}, {"type": "points", "data": [[0.0, 0.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[0.0, 0.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[0.0, 0.0, 1.0]], "layers": [4]}, {"type": "points", "data": [[0.0, 0.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[3.0, 0.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[3.0, 0.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[3.0, 0.0, 1.0]], "layers": [4]}, {"type": "points", "data": [[3.0, 0.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[8.0, 0.0, 0.0]], "layers": [4]}]}

