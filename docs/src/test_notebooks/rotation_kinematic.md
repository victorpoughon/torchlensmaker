# Rotation on the kinematic chain


```python
import torch
import torchlensmaker as tlm
import torch.optim as optim

surface = tlm.Parabola(diameter=15, A=tlm.parameter(0.01))

mirror = tlm.ReflectiveSurface(tlm.SquarePlane(20))

lens = tlm.BiLens(surface, material = 'BK7-nd', outer_thickness=1.0)

optics = tlm.Sequential(
    tlm.PointSourceAtInfinity(beam_diameter=18.5),
    tlm.Gap(10),
    lens,
    
    tlm.Gap(30),
    tlm.SubChain(
        tlm.MixedDim(dim2=tlm.Rotate2D(45), dim3=tlm.Rotate3D(y=45, z=0)),
        mirror
    ),

    tlm.MixedDim(dim2=tlm.Rotate2D(-90), dim3=tlm.Rotate3D(y=-90, z=0)),
    tlm.Gap(30),
    tlm.FocalPoint(),
)

tlm.show2d(optics)
tlm.show3d(optics, sampling={"base":100})
```


<TLMViewer src="./rotation_kinematic_files/rotation_kinematic_0.json?url" />



<TLMViewer src="./rotation_kinematic_files/rotation_kinematic_1.json?url" />

