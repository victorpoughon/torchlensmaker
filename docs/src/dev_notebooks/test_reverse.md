```python
# Test reverse()

import torch
import torchlensmaker as tlm

doublet = tlm.Sequential(
    tlm.RefractiveSurface(tlm.Sphere(4.0, C=0.135327), material=tlm.NonDispersiveMaterial(1.517)),
    tlm.Gap(1.05),
    tlm.RefractiveSurface(tlm.Sphere(4.0, C=-0.19311), material=tlm.NonDispersiveMaterial(1.649)),
    tlm.Gap(0.4),
    tlm.RefractiveSurface(tlm.Sphere(4.0, C=-0.06164), material="air"),
)

def show2d_with_light(lens):
    optics = tlm.Sequential(
        tlm.PointSourceAtInfinity(3.0),
        tlm.Gap(2),
        doublet,
        #tlm.Gap(2), # use an element for end=?
    )

    optics.set_sampling2d(pupil=20)
    tlm.show(optics, dim=2, controls={"show_optical_axis": True, "show_other_axes": True}, end=2)

doublet.reverse()

tlm.show2d(doublet, controls={"show_optical_axis": True, "show_other_axes": True})
tlm.show2d(doublet.reverse(), controls={"show_optical_axis": True, "show_other_axes": True})
```


<TLMViewer src="./test_reverse_files/test_reverse_0.json?url" />



<TLMViewer src="./test_reverse_files/test_reverse_1.json?url" />

