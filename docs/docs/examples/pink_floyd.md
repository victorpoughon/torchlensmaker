# Pink Floyd


```python
import torch
import torchlensmaker as tlm
import math

S = 5
R = S/2
A = 30
gamma = 60

optics = tlm.Sequential(
    tlm.Turn([20, 0]),
    tlm.RaySource(material="air"),
    tlm.Wavelength(400, 700),
    tlm.Gap(10),
    tlm.Turn([-20, 0]),
    tlm.Rotate(
        tlm.RefractiveSurface(tlm.CircularPlane(S), material="BK7"),
        [-A, 0]),
    tlm.Gap(R),
    tlm.Rotate(
        tlm.RefractiveSurface(tlm.CircularPlane(S), material="air", critical_angle="clamp"),
        [A, 0]),
)

sampling = {"wavelength": 10}

output = optics(tlm.default_input(dim=2, dtype=torch.float64, sampling=sampling))

tlm.show(optics, dim=2, end=10, sampling=sampling)
tlm.show(optics, dim=3, end=10, sampling=sampling)
```


<TLMViewer src="./pink_floyd_tlmviewer/pink_floyd_0.json" />



<TLMViewer src="./pink_floyd_tlmviewer/pink_floyd_1.json" />

