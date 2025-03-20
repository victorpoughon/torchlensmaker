# Magnifying glass


```python
import torch
import torchlensmaker as tlm

surface = tlm.Sphere(diameter=30, R=50)

optics = tlm.Sequential(
    tlm.Object(beam_angular_size=10, object_diameter=10),
    tlm.Gap(20),
    tlm.RefractiveSurface(surface, anchors=("origin", "extent"), material="BK7-nd"),
    tlm.Gap(3),
    tlm.RefractiveSurface(surface, scale=-1, anchors=("extent", "origin"), material="air"),
    tlm.Gap(-44.5),
    tlm.ImagePlane(50),
)

tlm.show(optics, dim=2)
```


<TLMViewer src="./magnifying_glass_files/magnifying_glass_0.json?url" />

