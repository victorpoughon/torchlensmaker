# Biconvex Parabolic Lens


```python
import torch
import torchlensmaker as tlm
import torch.optim as optim

# y = a*x^2
surface = tlm.Parabola(diameter=15, A=tlm.parameter(0.015))

lens = tlm.BiLens(surface, material="BK7", outer_thickness=1.0)

optics = tlm.Sequential(
    tlm.PointSourceAtInfinity(beam_diameter=18.5),
    tlm.Wavelength(500, 800),
    tlm.Gap(10),
    lens,
    tlm.Gap(50),
    tlm.FocalPoint(),
)

tlm.show(optics, dim=2, sampling={"base": 10, "wavelength": 10})
```


<TLMViewer src="./biconvex_parabola_files/biconvex_parabola_0.json?url" />



```python
tlm.optimize(
    optics,
    optimizer = optim.Adam(optics.parameters(), lr=3e-4),
    sampling = {"base": 10, "wavelength": 10},
    dim = 2,
    num_iter = 100
).plot()

print("Final parabola parameter:", surface.sag_function.A.item())
print("Outer thickness:", lens.outer_thickness())
print("Inner thickness:", lens.inner_thickness())
```

    [  1/100] L= 2.44664 | grad norm= 452.9618884131326
    [  6/100] L= 1.77205 | grad norm= 449.25156671386145
    [ 11/100] L= 1.10307 | grad norm= 445.83216393762723
    [ 16/100] L= 0.43969 | grad norm= 442.6732215467563
    [ 21/100] L= 0.18247 | grad norm= 439.90324532114613
    [ 26/100] L= 0.34463 | grad norm= 439.2101957117343
    [ 31/100] L= 0.15203 | grad norm= 440.0346400093066
    [ 36/100] L= 0.14041 | grad norm= 441.31836988080266
    [ 41/100] L= 0.07994 | grad norm= 425.14869121288865
    [ 46/100] L= 0.09076 | grad norm= 440.300364325908
    [ 51/100] L= 0.04187 | grad norm= 176.87097772671547
    [ 56/100] L= 0.03977 | grad norm= 155.25726417068657
    [ 61/100] L= 0.04401 | grad norm= 254.642424997515
    [ 66/100] L= 0.04265 | grad norm= 204.12878776101755
    [ 71/100] L= 0.03820 | grad norm= 176.18179004686544
    [ 76/100] L= 0.03618 | grad norm= 100.56780990442584
    [ 81/100] L= 0.03462 | grad norm= 70.39112718943763
    [ 86/100] L= 0.03467 | grad norm= 51.459174495373595
    [ 91/100] L= 0.03432 | grad norm= 12.458666839218564
    [ 96/100] L= 0.03467 | grad norm= 51.459143670865416
    [100/100] L= 0.03434 | grad norm= 12.460434795945261



    
![png](biconvex_parabola_files/biconvex_parabola_2_1.png)
    


    Final parabola parameter: 0.009488203272067224
    Outer thickness: tensor(1.0000, dtype=torch.float64, grad_fn=<LinalgVectorNormBackward0>)
    Inner thickness: tensor(2.0674, dtype=torch.float64, grad_fn=<LinalgVectorNormBackward0>)



```python
tlm.show(optics, dim=2, end=60)
tlm.show(optics, dim=3, sampling={"object": 10, "base": 64, "wavelength": 5}, end=60)
```


<TLMViewer src="./biconvex_parabola_files/biconvex_parabola_1.json?url" />



<TLMViewer src="./biconvex_parabola_files/biconvex_parabola_2.json?url" />



```python
part = tlm.export.lens_to_part(lens)
tlm.show_part(part)
```


<em>part display not supported in vitepress</em>

