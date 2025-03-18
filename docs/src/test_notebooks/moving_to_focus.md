# Moving a lens to focus


```python
import torch
import torch.nn as nn
import torchlensmaker as tlm
import torch.optim as optim

surface = tlm.Parabola(diameter=15, A=0.006)

x = tlm.parameter(50)

optics = nn.Sequential(
    tlm.PointSourceAtInfinity(beam_diameter=18.5),
    tlm.Gap(10),
    tlm.RefractiveSurface(surface, material="BK7-nd", anchors=("origin", "extent")),
    tlm.Gap(2),
    tlm.RefractiveSurface(
        surface, material="air", scale=-1, anchors=("extent", "origin")
    ),
    tlm.Gap(x),
    tlm.FocalPoint(),
)


tlm.show(optics, dim=2)
tlm.show(optics, dim=3)
```


<TLMViewer src="./moving_to_focus_files/moving_to_focus_0.json?url" />



<TLMViewer src="./moving_to_focus_files/moving_to_focus_1.json?url" />



```python
tlm.optimize(
    optics,
    optimizer = optim.Adam(optics.parameters(), lr=.8),
    sampling = {"base": 10},
    dim = 2,
    num_iter = 100
).plot()
```

    [  1/100] L=  1.490 | grad norm= 0.0512464992118436
    [  6/100] L=  1.285 | grad norm= 0.0512464992118436
    [ 11/100] L=  1.080 | grad norm= 0.0512464992118436
    [ 16/100] L=  0.875 | grad norm= 0.0512464992118436
    [ 21/100] L=  0.670 | grad norm= 0.0512464992118436
    [ 26/100] L=  0.465 | grad norm= 0.0512464992118436
    [ 31/100] L=  0.260 | grad norm= 0.0512464992118436
    [ 36/100] L=  0.055 | grad norm= 0.0512464992118436
    [ 41/100] L=  0.103 | grad norm= 0.0512464992118436
    [ 46/100] L=  0.117 | grad norm= 0.0512464992118436
    [ 51/100] L=  0.040 | grad norm= 0.0512464992118436
    [ 56/100] L=  0.052 | grad norm= 0.0512464992118436
    [ 61/100] L=  0.032 | grad norm= 0.0512464992118436
    [ 66/100] L=  0.036 | grad norm= 0.0512464992118436
    [ 71/100] L=  0.013 | grad norm= 0.025671685539087233
    [ 76/100] L=  0.023 | grad norm= 0.0512464992118436
    [ 81/100] L=  0.017 | grad norm= 0.025671685539087233
    [ 86/100] L=  0.012 | grad norm= 0.006342011105652315
    [ 91/100] L=  0.013 | grad norm= 0.006342011105652315
    [ 96/100] L=  0.013 | grad norm= 0.025671685539087233
    [100/100] L=  0.013 | grad norm= 0.006342011105652315



    
![png](moving_to_focus_files/moving_to_focus_2_1.png)
    



```python
tlm.show(optics, dim=2)
```


<TLMViewer src="./moving_to_focus_files/moving_to_focus_2.json?url" />

