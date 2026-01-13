# Planoconvex spherical lens


```python
import torch
import torchlensmaker as tlm
import torch.optim as optim

surface = tlm.Parabola(diameter=15, A=tlm.parameter(0.03))

lens = tlm.PlanoLens(surface, material = 'BK7-nd', outer_thickness=1.0, reverse=True)

optics = tlm.Sequential(
    tlm.PointSourceAtInfinity(beam_diameter=18.5),
    tlm.Gap(10),
    lens,
    tlm.Gap(50),
    tlm.FocalPoint(),
)

tlm.show(optics, dim=2)
tlm.show(optics, dim=3)
```


<TLMViewer src="./planoconvex_sphere_files/planoconvex_sphere_0.json?url" />



<TLMViewer src="./planoconvex_sphere_files/planoconvex_sphere_1.json?url" />



```python
tlm.optimize(
    optics,
    optimizer = optim.Adam(optics.parameters(), lr=3e-4),
    sampling = {"base": 10},
    dim = 2,
    num_iter = 100
).plot()

print("Final parabola parameter:", surface.A.item())
print("Outer thickness:", lens.outer_thickness())
print("Inner thickness:", lens.inner_thickness())
```

    [  1/100] L= 10.37320 | grad norm= 196.33741603026206
    [  6/100] L= 10.07675 | grad norm= 197.8302807867228
    [ 11/100] L= 9.77791 | grad norm= 199.28785472302678
    [ 16/100] L= 9.47657 | grad norm= 200.70862303431198
    [ 21/100] L= 9.17264 | grad norm= 202.09085631779132
    [ 26/100] L= 8.86605 | grad norm= 203.4326108056964
    [ 31/100] L= 8.55678 | grad norm= 204.73175176408733
    [ 36/100] L= 8.24484 | grad norm= 205.98599311084854
    [ 41/100] L= 7.93027 | grad norm= 207.1929453403303
    [ 46/100] L= 7.61314 | grad norm= 208.35016514355746
    [ 51/100] L= 7.29354 | grad norm= 209.45520238779704
    [ 56/100] L= 6.97157 | grad norm= 210.50564230482328
    [ 61/100] L= 6.64734 | grad norm= 211.49914230894558
    [ 66/100] L= 6.32101 | grad norm= 212.43346374931284
    [ 71/100] L= 5.99270 | grad norm= 213.3064992518031
    [ 76/100] L= 5.66257 | grad norm= 214.1162963318701
    [ 81/100] L= 5.33077 | grad norm= 214.8610778316225
    [ 86/100] L= 4.99749 | grad norm= 215.5392595620278
    [ 91/100] L= 4.66288 | grad norm= 216.1494653725264
    [ 96/100] L= 4.32711 | grad norm= 216.69053974909636
    [100/100] L= 4.05780 | grad norm= 217.07299523664332



    
![png](planoconvex_sphere_files/planoconvex_sphere_2_1.png)
    


    Final parabola parameter: -0.0005561670680468251
    Outer thickness: tensor(1., dtype=torch.float64, grad_fn=<LinalgVectorNormBackward0>)
    Inner thickness: tensor(1.0313, dtype=torch.float64, grad_fn=<LinalgVectorNormBackward0>)



```python
tlm.show(optics, dim=2, end=60)
tlm.show(optics, dim=3, end=60)
```


<TLMViewer src="./planoconvex_sphere_files/planoconvex_sphere_2.json?url" />



<TLMViewer src="./planoconvex_sphere_files/planoconvex_sphere_3.json?url" />



```python
part = tlm.export.lens_to_part(lens)
tlm.show_part(part)
```


<em>part display not supported in vitepress</em>

