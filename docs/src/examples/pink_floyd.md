# Pink Floyd


```python
import torch
import torchlensmaker as tlm
import math

S = 5
R = S/2
A = 30

optics = tlm.Sequential(
    tlm.Rotate2D(20),
    tlm.RaySource(material="air", wavelength=(400, 700)),
    tlm.Gap(10),
    tlm.Rotate2D(-20),
    tlm.SubChain(
        tlm.Rotate2D(-A),
        tlm.RefractiveSurface(tlm.CircularPlane(S), material="K5"),
    ),
    tlm.Gap(R),
    tlm.SubChain(
        tlm.Rotate2D(A),
        tlm.RefractiveSurface(tlm.CircularPlane(S), material="air"),
    )
)

optics.set_sampling2d(wavelength=10)

output = optics(tlm.default_input(dim=2))

tlm.show(optics, dim=2, end=10, controls={"color_rays": "wavelength"})
```


<TLMViewer src="./pink_floyd_files/pink_floyd_0.json?url" />

