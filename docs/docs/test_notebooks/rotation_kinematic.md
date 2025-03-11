# Rotation on the kinematic chain


```python
import torch
import torch.nn as nn
import torchlensmaker as tlm
import torch.optim as optim

surface = tlm.Parabola(diameter=15, A=tlm.parameter(0.01))

mirror = tlm.ReflectiveSurface(tlm.SquarePlane(20))

lens = tlm.BiLens(surface, material = 'BK7-nd', outer_thickness=1.0)

optics = nn.Sequential(
    tlm.PointSourceAtInfinity(beam_diameter=18.5),
    tlm.Gap(10),
    lens,
    
    tlm.Gap(30),
    tlm.Rotate(mirror, (45, 0)),

    tlm.Turn((-90, 0)),
    tlm.Gap(30),
    tlm.FocalPoint(),
)

tlm.show(optics, dim=2)
tlm.show(optics, dim=3, sampling={"base":100})
```


<TLMViewer src="./rotation_kinematic_tlmviewer/rotation_kinematic_0.json" />



<TLMViewer src="./rotation_kinematic_tlmviewer/rotation_kinematic_1.json" />

