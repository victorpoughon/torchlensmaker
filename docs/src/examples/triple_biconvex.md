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

    [  1/100] L= 1.31674 | grad norm= 954.9683837890625
    [  6/100] L= 0.83680 | grad norm= 968.0955810546875
    [ 11/100] L= 0.34967 | grad norm= 980.7728881835938
    [ 16/100] L= 0.11949 | grad norm= 992.4049682617188
    [ 21/100] L= 0.22138 | grad norm= 994.8600463867188
    [ 26/100] L= 0.05023 | grad norm= 990.7224731445312
    [ 31/100] L= 0.10033 | grad norm= 987.02294921875
    [ 36/100] L= 0.02034 | grad norm= 269.2731628417969
    [ 41/100] L= 0.04776 | grad norm= 990.6621704101562
    [ 46/100] L= 0.05121 | grad norm= 988.236328125
    [ 51/100] L= 0.03579 | grad norm= 990.369873046875
    [ 56/100] L= 0.03184 | grad norm= 910.9385986328125
    [ 61/100] L= 0.01983 | grad norm= 269.2564697265625
    [ 66/100] L= 0.01662 | grad norm= 284.94482421875
    [ 71/100] L= 0.01564 | grad norm= 284.9621887207031
    [ 76/100] L= 0.01862 | grad norm= 269.2178039550781
    [ 81/100] L= 0.01869 | grad norm= 284.9084777832031
    [ 86/100] L= 0.01697 | grad norm= 269.16546630859375
    [ 91/100] L= 0.01726 | grad norm= 269.1744079589844
    [ 96/100] L= 0.01724 | grad norm= 269.1739807128906
    [100/100] L= 0.01769 | grad norm= 284.9261779785156



    
![png](triple_biconvex_files/triple_biconvex_2_1.png)
    



```python
print("Final parabola parameter:", surface.sag_function.A.item())
print("Outer thickness:", lens1.outer_thickness().item())
print("Inner thickness:", lens1.inner_thickness().item())
```

    Final parabola parameter: 0.003654894884675741
    Outer thickness: 0.5
    Inner thickness: 0.9111757278442383



```python
tlm.show(optics, dim=2)
tlm.show(optics, dim=3)
```


<TLMViewer src="./triple_biconvex_files/triple_biconvex_2.json?url" />



<TLMViewer src="./triple_biconvex_files/triple_biconvex_3.json?url" />

