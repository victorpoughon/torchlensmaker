# Concave mirror


```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchlensmaker as tlm

surface = tlm.Parabola(diameter=35.0, A=tlm.parameter(-0.002))

optics = nn.Sequential(
    tlm.PointSourceAtInfinity(beam_diameter=25),
    tlm.Gap(100),
    tlm.ReflectiveSurface(surface),
    tlm.Gap(-50),
    tlm.FocalPoint(),
)

tlm.show(optics, dim=2)
tlm.show(optics, dim=3)
```


<TLMViewer src="./concave_mirror_files/concave_mirror_0.json?url" />



<TLMViewer src="./concave_mirror_files/concave_mirror_1.json?url" />



```python
tlm.optimize(
    optics,
    optimizer = optim.Adam(optics.parameters(), lr=1e-4),
    sampling = {"base": 10},
    dim = 2,
    num_iter = 100
).plot()
```

    [  1/100] L=  4.160 | grad norm= 1394.1171077050594
    [  6/100] L=  3.464 | grad norm= 1393.4490833596467
    [ 11/100] L=  2.768 | grad norm= 1391.9976497420437
    [ 16/100] L=  2.074 | grad norm= 1389.7699097324544
    [ 21/100] L=  1.381 | grad norm= 1386.7754449116364
    [ 26/100] L=  0.691 | grad norm= 1383.026448515728
    [ 31/100] L=  0.003 | grad norm= 1378.5377698947361
    [ 36/100] L=  0.428 | grad norm= 1375.3431857750766
    [ 41/100] L=  0.394 | grad norm= 1375.600375207619
    [ 46/100] L=  0.092 | grad norm= 1377.8558255379692
    [ 51/100] L=  0.213 | grad norm= 1379.9900408064798
    [ 56/100] L=  0.129 | grad norm= 1379.4181478274747
    [ 61/100] L=  0.122 | grad norm= 1377.640168120014
    [ 66/100] L=  0.045 | grad norm= 1378.1925690680116
    [ 71/100] L=  0.032 | grad norm= 1378.744133961983
    [ 76/100] L=  0.049 | grad norm= 1378.1684932842547
    [ 81/100] L=  0.012 | grad norm= 1378.6024576988589
    [ 86/100] L=  0.006 | grad norm= 1378.559737394575
    [ 91/100] L=  0.007 | grad norm= 1378.5630411496224
    [ 96/100] L=  0.030 | grad norm= 1378.7298983396522
    [100/100] L=  0.002 | grad norm= 1378.5323313003391



    
![png](concave_mirror_files/concave_mirror_2_1.png)
    



```python
tlm.show(optics, dim=2)
tlm.show(optics, dim=3)
```


<TLMViewer src="./concave_mirror_files/concave_mirror_2.json?url" />



<TLMViewer src="./concave_mirror_files/concave_mirror_3.json?url" />

