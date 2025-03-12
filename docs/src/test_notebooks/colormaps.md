```python
import numpy as np
import colorcet as cc

def export_cmap(cmap):
    _ = cmap(0.5)
    lut = cmap._lut[:256, :3]

    print(lut.tolist())

#export_cmap(cc.cm.CET_I2)
#export_cmap(cc.cm.CET_R4)
```


```python
import torch
import torchlensmaker as tlm

S = torch.linspace(-10, 10, 150)

start = torch.stack((S, torch.zeros_like(S)), dim=-1)
end = torch.stack((3*S + 1, torch.full_like(S, 15)), dim=-1)

variables = {
    "var1": S,
    "var2": S/2,
    "var3": S/3,
    "var-over": 2*S,
}

domain = {
    "var1": [-10, 10],
    "var2": [-10, 10],
    "var3": [-10, 10],
    "var-over": [-10, 10],
}

scene = tlm.viewer.new_scene("2D")
scene["data"].append(tlm.viewer.render_rays(start, end, 0, variables, domain))
tlm.viewer.display_scene(scene)
```


<TLMViewer src="./colormaps_tlmviewer/colormaps_0.json?url" />

