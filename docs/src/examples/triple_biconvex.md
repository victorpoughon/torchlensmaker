# Triple Lens


```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchlensmaker as tlm

lens_diameter = 15.0

surface = tlm.Parabola(lens_diameter, A=tlm.parameter(0.005))
lens1 = tlm.lenses.symmetric_singlet(surface, tlm.OuterGap(0.5), material = 'BK7')
lens3 = tlm.lenses.symmetric_singlet(surface, tlm.OuterGap(0.5), material = 'BK7')
lens2 = tlm.lenses.symmetric_singlet(surface, tlm.OuterGap(0.5), material = 'BK7')

optics = tlm.Sequential(
    tlm.PointSourceAtInfinity(0.9*lens_diameter),
    tlm.Gap(15),
    
    lens1,
    tlm.Gap(5),
    lens2,
    tlm.Gap(5),
    lens3,
    
    tlm.Gap(40),
    tlm.FocalPoint(),
)

tlm.show(optics, dim=2)
tlm.show(optics, dim=3)
```


<TLMViewer src="./triple_biconvex_files/triple_biconvex_0.json?url" />



<TLMViewer src="./triple_biconvex_files/triple_biconvex_1.json?url" />



```python
optics.set_sampling2d(pupil=10)
tlm.optimize(
    optics,
    optimizer = optim.Adam(optics.parameters(), lr=1e-4),
    dim = 2,
    num_iter = 100
).plot()
```

    [  1/100] L= 1.31674 | grad norm= 953.393798828125
    [  6/100] L= 0.83680 | grad norm= 966.7118530273438
    [ 11/100] L= 0.34966 | grad norm= 979.573486328125
    [ 16/100] L= 0.11953 | grad norm= 991.3750610351562
    [ 21/100] L= 0.22137 | grad norm= 993.8638305664062
    [ 26/100] L= 0.05014 | grad norm= 989.6644287109375
    [ 31/100] L= 0.10048 | grad norm= 985.91015625
    [ 36/100] L= 0.02030 | grad norm= 268.8476257324219
    [ 41/100] L= 0.04759 | grad norm= 989.6012573242188
    [ 46/100] L= 0.05139 | grad norm= 987.1400146484375
    [ 51/100] L= 0.03561 | grad norm= 989.304931640625
    [ 56/100] L= 0.03201 | grad norm= 909.9578247070312
    [ 61/100] L= 0.01978 | grad norm= 268.830810546875
    [ 66/100] L= 0.01666 | grad norm= 284.7691650390625
    [ 71/100] L= 0.01566 | grad norm= 284.7873229980469
    [ 76/100] L= 0.01862 | grad norm= 268.7931823730469
    [ 81/100] L= 0.01868 | grad norm= 284.7330017089844
    [ 86/100] L= 0.01701 | grad norm= 268.7413330078125
    [ 91/100] L= 0.01731 | grad norm= 268.7510986328125
    [ 96/100] L= 0.01585 | grad norm= 284.7837829589844
    [100/100] L= 0.01680 | grad norm= 284.76666259765625



    
![png](triple_biconvex_files/triple_biconvex_2_1.png)
    



```python
print("Final parabola parameter:", surface.A.item())
print("Outer thickness:", lens1.outer_thickness().item())
print("Inner thickness:", lens1.inner_thickness().item())
```

    Final parabola parameter: 0.0036443897988647223
    Outer thickness: 0.5
    Inner thickness: 0.9099938869476318



```python
tlm.show(optics, dim=2)
tlm.show(optics, dim=3)
```


<TLMViewer src="./triple_biconvex_files/triple_biconvex_2.json?url" />



<TLMViewer src="./triple_biconvex_files/triple_biconvex_3.json?url" />

