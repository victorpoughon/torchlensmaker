# Bounding cylinder of sag surfaces


```python
import torchlensmaker as tlm
import torch
import json

sphere = tlm.Sphere(10, R=6.5)
parabola = tlm.Parabola(10, A=-0.2, normalize=True)
conic = tlm.SagSurface(10, tlm.Conical(C=torch.tensor(1.0), K=torch.tensor(-4.5), normalize=True))
aspheric1 = tlm.SagSurface(10, tlm.Aspheric(coefficients=torch.tensor([1, 0.0, 0.0]), normalize=True))
aspheric2 = tlm.SagSurface(10, tlm.Aspheric(coefficients=torch.tensor([0.0, -1, 0.5]), normalize=True))
aspheric3 = tlm.SagSurface(10, tlm.Aspheric(coefficients=torch.tensor([0.0, 0.0, 1]), normalize=True))

sum0 = tlm.SagSurface(10, tlm.SagSum([
    tlm.Aspheric(coefficients=torch.tensor([0.0, -1, 0.5]), normalize=True),
    tlm.Conical(C=torch.tensor(1.0), K=torch.tensor(-4.5), normalize=True)
]))

optics = tlm.Sequential(
    tlm.Gap(5),
    tlm.ReflectiveSurface(sphere),
    tlm.Gap(5),
    
    tlm.SubChain(
        tlm.MixedDim(tlm.Rotate2D(15), tlm.Rotate3D(y=15, z=-10)),  
        tlm.ReflectiveSurface(parabola)
    ),
    
    tlm.Gap(5),
    tlm.SubChain(
        tlm.MixedDim(tlm.Rotate2D(-15), tlm.Rotate3D(y=-15, z=10)),  
        tlm.ReflectiveSurface(conic)
    ),
    tlm.Gap(5),

    tlm.SubChain(
        tlm.MixedDim(tlm.Rotate2D(25), tlm.Rotate3D(y=25, z=-10)),
        tlm.ReflectiveSurface(aspheric2),
    ),
    
    tlm.Gap(5),
    tlm.ReflectiveSurface(sum0),
)

scene2d = tlm.show2d(optics, controls={"show_optical_axis": True, "show_bounding_cylinders": True}, return_scene=True)
scene3d = tlm.show3d(optics, controls={"show_optical_axis": True, "show_bounding_cylinders": True}, return_scene=True)

print(json.dumps(scene2d))
print(json.dumps(scene3d))
```


<TLMViewer src="./bounding_cylinder_files/bounding_cylinder_0.json?url" />



<TLMViewer src="./bounding_cylinder_files/bounding_cylinder_1.json?url" />


    {"mode": "2D", "camera": "XY", "data": [{"type": "surface-sag", "diameter": 10, "sag-function": {"sag-type": "spherical", "C": 0.15384615384615385}, "bcyl": [0.0, 2.3466880685409626, 5.0], "matrix": [[1.0, 0.0, 5.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]}, {"type": "surface-sag", "diameter": 10, "sag-function": {"sag-type": "parabolic", "A": -0.2, "normalize": true}, "bcyl": [-1.0, 0.0, 5.0], "matrix": [[0.9659258127212524, -0.258819043636322, 10.0], [0.258819043636322, 0.9659258127212524, 0.0], [0.0, 0.0, 1.0]]}, {"type": "surface-sag", "diameter": 10, "sag-function": {"sag-type": "conical", "K": -4.5, "C": 1.0, "normalize": true}, "bcyl": [0.0, 1.6018862050852036, 5.0], "matrix": [[0.9659258127212524, 0.258819043636322, 15.0], [-0.258819043636322, 0.9659258127212524, 0.0], [0.0, 0.0, 1.0]]}, {"type": "surface-sag", "diameter": 10, "sag-function": {"sag-type": "aspheric", "coefficients": [0.0, -1.0, 0.5], "normalize": true}, "bcyl": [-5.0, 2.5, 5.0], "matrix": [[0.9063078165054321, -0.4226182699203491, 20.0], [0.4226182699203491, 0.9063078165054321, 0.0], [0.0, 0.0, 1.0]]}, {"type": "surface-sag", "diameter": 10, "sag-function": {"sag-type": "sum", "terms": [{"sag-type": "aspheric", "coefficients": [0.0, -1.0, 0.5], "normalize": true}, {"sag-type": "conical", "K": -4.5, "C": 1.0, "normalize": true}]}, "bcyl": [-5.0, 4.101886205085203, 5.0], "matrix": [[1.0, 0.0, 25.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]}, {"type": "points", "data": [[0.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[5.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[5.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[10.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[10.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[10.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[15.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[15.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[15.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[20.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[20.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[20.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[25.0, 0.0]], "layers": [4]}], "controls": {"show_optical_axis": true, "show_bounding_cylinders": true}}
    {"mode": "3D", "camera": "orthographic", "data": [{"type": "surface-sag", "diameter": 10, "sag-function": {"sag-type": "spherical", "C": 0.15384615384615385}, "bcyl": [0.0, 2.3466880685409626, 5.0], "matrix": [[1.0, 0.0, 0.0, 5.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]}, {"type": "surface-sag", "diameter": 10, "sag-function": {"sag-type": "parabolic", "A": -0.2, "normalize": true}, "bcyl": [-1.0, 0.0, 5.0], "matrix": [[0.9512512415589065, 0.1677312543393222, 0.25881905213951417, 10.0], [-0.17364817266677937, 0.9848077538938695, 0.0, 0.0], [-0.2548870094024553, -0.04494345545537453, 0.9659258244035116, 0.0], [0.0, 0.0, 0.0, 1.0]]}, {"type": "surface-sag", "diameter": 10, "sag-function": {"sag-type": "conical", "K": -4.5, "C": 1.0, "normalize": true}, "bcyl": [0.0, 1.6018862050852036, 5.0], "matrix": [[0.9512512415589065, -0.1677312543393222, -0.25881905213951417, 15.0], [0.17364817266677937, 0.9848077538938695, 0.0, 0.0], [0.2548870094024553, -0.04494345545537453, 0.9659258244035116, 0.0], [0.0, 0.0, 0.0, 1.0]]}, {"type": "surface-sag", "diameter": 10, "sag-function": {"sag-type": "aspheric", "coefficients": [0.0, -1.0, 0.5], "normalize": true}, "bcyl": [-5.0, 2.5, 5.0], "matrix": [[0.8925389351691448, 0.15737869093055265, 0.42261826374177747, 20.0], [-0.17364817266677937, 0.9848077538938695, 0.0, 0.0], [-0.41619774307006685, -0.07338688923436668, 0.9063077861035319, 0.0], [0.0, 0.0, 0.0, 1.0]]}, {"type": "surface-sag", "diameter": 10, "sag-function": {"sag-type": "sum", "terms": [{"sag-type": "aspheric", "coefficients": [0.0, -1.0, 0.5], "normalize": true}, {"sag-type": "conical", "K": -4.5, "C": 1.0, "normalize": true}]}, "bcyl": [-5.0, 4.101886205085203, 5.0], "matrix": [[1.0, 0.0, 0.0, 25.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]}, {"type": "points", "data": [[0.0, 0.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[5.0, 0.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[5.0, 0.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[10.0, 0.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[10.0, 0.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[10.0, 0.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[15.0, 0.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[15.0, 0.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[15.0, 0.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[20.0, 0.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[20.0, 0.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[20.0, 0.0, 0.0]], "layers": [4]}, {"type": "points", "data": [[25.0, 0.0, 0.0]], "layers": [4]}], "controls": {"show_optical_axis": true, "show_bounding_cylinders": true}}



```python
torch.sum(torch.tensor((), dtype=torch.float64))
```




    tensor(0., dtype=torch.float64)


