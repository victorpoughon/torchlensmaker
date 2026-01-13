# Triple Lens


```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchlensmaker as tlm

lens_diameter = 15.0

surface = tlm.Parabola(lens_diameter, A=tlm.parameter(0.005))
lens1 = tlm.BiLens(surface, material = 'BK7-nd', outer_thickness=0.5)
lens2 = tlm.BiLens(surface, material = 'BK7-nd', outer_thickness=0.5)
lens3 = tlm.BiLens(surface, material = 'BK7-nd', outer_thickness=0.5)

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
tlm.optimize(
    optics,
    optimizer = optim.Adam(optics.parameters(), lr=1e-4),
    sampling = {"base": 10},
    dim = 2,
    num_iter = 100
).plot()
```

    [  1/100] L= 1.31674 | grad norm= 954.968294573738
    [  6/100] L= 0.83680 | grad norm= 968.0955854012096
    [ 11/100] L= 0.34967 | grad norm= 980.7726712666237
    [ 16/100] L= 0.11950 | grad norm= 992.4052121101311
    [ 21/100] L= 0.22138 | grad norm= 994.8600417065833
    [ 26/100] L= 0.05024 | grad norm= 990.7222479033197
    [ 31/100] L= 0.10033 | grad norm= 987.0230588519908
    [ 36/100] L= 0.02034 | grad norm= 269.27305220288434
    [ 41/100] L= 0.04776 | grad norm= 990.6618967844837
    [ 46/100] L= 0.05121 | grad norm= 988.2361085642427
    [ 51/100] L= 0.03579 | grad norm= 990.3697737864526
    [ 56/100] L= 0.03184 | grad norm= 910.9385309052975
    [ 61/100] L= 0.01983 | grad norm= 269.25652762953536
    [ 66/100] L= 0.01662 | grad norm= 284.9449255943511
    [ 71/100] L= 0.01564 | grad norm= 284.96223131970004
    [ 76/100] L= 0.01862 | grad norm= 269.2179981572597
    [ 81/100] L= 0.01869 | grad norm= 284.9085496927424
    [ 86/100] L= 0.01697 | grad norm= 269.16543199722395
    [ 91/100] L= 0.01726 | grad norm= 269.1744179024867
    [ 96/100] L= 0.01724 | grad norm= 269.1739697985022
    [100/100] L= 0.01769 | grad norm= 284.9262002640567



    
![png](triple_biconvex_files/triple_biconvex_2_1.png)
    



```python
print("Final parabola parameter:", surface.sag_function.A.item())
print("Outer thickness:", lens1.outer_thickness().item())
print("Inner thickness:", lens1.inner_thickness().item())
```

    Final parabola parameter: 0.0036548946158946652
    Outer thickness: 0.5
    Inner thickness: 0.9111756442881498



```python
tlm.show(optics, dim=2)
tlm.show(optics, dim=3)
```


<TLMViewer src="./triple_biconvex_files/triple_biconvex_2.json?url" />



<TLMViewer src="./triple_biconvex_files/triple_biconvex_3.json?url" />

