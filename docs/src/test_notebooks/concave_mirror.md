# Concave mirror


```python
import torch
import torch.optim as optim
import torchlensmaker as tlm

surface = tlm.Parabola(diameter=35.0, A=tlm.parameter(-0.002))

optics = tlm.Sequential(
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

    [  1/100] L= 4.16038 | grad norm= 1394.1171077050594
    [  6/100] L= 3.46405 | grad norm= 1393.4490833596467
    [ 11/100] L= 2.76842 | grad norm= 1391.997649742044
    [ 16/100] L= 2.07398 | grad norm= 1389.7699097324544
    [ 21/100] L= 1.38124 | grad norm= 1386.7754449116364
    [ 26/100] L= 0.69077 | grad norm= 1383.026448515728
    [ 31/100] L= 0.00312 | grad norm= 1378.5377698947361
    [ 36/100] L= 0.42770 | grad norm= 1375.3431857750766
    [ 41/100] L= 0.39443 | grad norm= 1375.6003752076188
    [ 46/100] L= 0.09227 | grad norm= 1377.855825537969
    [ 51/100] L= 0.21338 | grad norm= 1379.9900408064798
    [ 56/100] L= 0.12936 | grad norm= 1379.4181478274747
    [ 61/100] L= 0.12202 | grad norm= 1377.640168120014
    [ 66/100] L= 0.04542 | grad norm= 1378.1925690680116
    [ 71/100] L= 0.03238 | grad norm= 1378.744133961983
    [ 76/100] L= 0.04879 | grad norm= 1378.1684932842547
    [ 81/100] L= 0.01227 | grad norm= 1378.602457698859
    [ 86/100] L= 0.00622 | grad norm= 1378.5597373945752
    [ 91/100] L= 0.00669 | grad norm= 1378.5630411496224
    [ 96/100] L= 0.03036 | grad norm= 1378.7298983396524
    [100/100] L= 0.00235 | grad norm= 1378.5323313003391



    
![png](concave_mirror_files/concave_mirror_2_1.png)
    



```python
tlm.show(optics, dim=2)
tlm.show(optics, dim=3)
```


<TLMViewer src="./concave_mirror_files/concave_mirror_2.json?url" />



<TLMViewer src="./concave_mirror_files/concave_mirror_3.json?url" />

