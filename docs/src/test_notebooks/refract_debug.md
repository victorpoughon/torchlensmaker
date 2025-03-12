```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchlensmaker as tlm

surface = tlm.Sphere(diameter=15, R=tlm.parameter(18))
lens = tlm.BiLens(surface, material="SF10-nd", outer_thickness=1.5)

optics = nn.Sequential(
    tlm.Offset(
        tlm.Rotate(
            tlm.RaySource(),
            [-55, 0]),
        y=17.9),
    tlm.Gap(10),
    lens,
    tlm.Gap(30),
    tlm.FocalPoint(),
)

lens.surface2.element[1].critical_angle = "clamp"

tlm.show(optics, dim=2, end=10, sampling={"base": 10})
```


<TLMViewer src="./refract_debug_tlmviewer/refract_debug_0.json?url" />



```python
### BUG V1
# happens no matter when the critical_angle argument is

import torch
import torch.nn as nn
import torch.optim as optim
import torchlensmaker as tlm


class Debug(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print(x.P.shape)
        return x

surface = tlm.Sphere(diameter=15, R=tlm.parameter(15))
lens = tlm.BiLens(surface, material="SF10-nd", outer_thickness=1.5)

optics = nn.Sequential(
    tlm.PointSourceAtInfinity(beam_diameter=18.5),
    Debug(),
    tlm.Gap(10),
    lens,
    Debug(),
    tlm.Gap(30),
    tlm.FocalPoint(),
)


lens.surface2.element[1].critical_angle = "drop"


tlm.show(optics, dim=2, sampling={"base": 10})
```

    torch.Size([10, 2])
    torch.Size([6, 2])



<TLMViewer src="./refract_debug_tlmviewer/refract_debug_1.json?url" />



```python
### BUG V2
# reproduction without lens

# bug:
# SF10-nd / clamp       : bug
# SF10-nd / drop        : bug maybe different
# SF10-nd / reflect     : unknown
# SF10-nd / nan         : json error

# no bug:
# water-nd / any

import torch
import torch.nn as nn
import torch.optim as optim
import torchlensmaker as tlm

surface = tlm.Sphere(diameter=15, R=-18)

optics = nn.Sequential(
    tlm.Offset(
        tlm.Rotate(
            tlm.RaySource(material="SF10-nd"),
            [-20, 0]),
        y=10),
    tlm.Gap(10),
    tlm.RefractiveSurface(surface, material="air", anchors=("origin", "origin"), critical_angle="clamp"),
    tlm.Gap(30),
    tlm.FocalPoint(),
)


tlm.show(optics, dim=2, sampling={"base": 10})
```


<TLMViewer src="./refract_debug_tlmviewer/refract_debug_2.json?url" />

