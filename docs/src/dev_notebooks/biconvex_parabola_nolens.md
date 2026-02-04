# Biconvex Parabolic Lens (no lens class)


```python
import torch
import torchlensmaker as tlm
import torch.optim as optim

surface = tlm.Parabola(diameter=15, A=tlm.parameter(0.02)) # y = a*x^2

optics = tlm.Sequential(
    tlm.PointSourceAtInfinity(beam_diameter=18.5),
    tlm.Gap(10),
    tlm.RefractiveSurface(surface, material="water", anchors=("origin", "extent")),
    tlm.Gap(2),
    tlm.RefractiveSurface(
        surface, material="water", scale=-1, anchors=("extent", "origin")
    ),
    tlm.Gap(50),
    tlm.FocalPoint(),
)

tlm.show(optics, dim=2)
tlm.show(optics, dim=3)
```


<TLMViewer src="./biconvex_parabola_nolens_files/biconvex_parabola_nolens_0.json?url" />



<TLMViewer src="./biconvex_parabola_nolens_files/biconvex_parabola_nolens_1.json?url" />



```python
optics.set_sampling2d(pupil=10)
tlm.optimize(
    optics,
    optimizer = optim.Adam(optics.parameters(), lr=3e-4),
    dim = 2,
    num_iter = 100
).plot()
```

    [  1/100] L= 1.88145 | grad norm= 113.95973205566406
    [  6/100] L= 1.71075 | grad norm= 114.18986511230469
    [ 11/100] L= 1.53972 | grad norm= 114.39652252197266
    [ 16/100] L= 1.36838 | grad norm= 114.58002471923828
    [ 21/100] L= 1.19673 | grad norm= 114.7408447265625
    [ 26/100] L= 1.02482 | grad norm= 114.87928771972656
    [ 31/100] L= 0.85267 | grad norm= 114.99583435058594
    [ 36/100] L= 0.68032 | grad norm= 115.09098052978516
    [ 41/100] L= 0.50781 | grad norm= 115.16510772705078
    [ 46/100] L= 0.33519 | grad norm= 115.21874237060547
    [ 51/100] L= 0.16250 | grad norm= 115.25240325927734
    [ 56/100] L= 0.07750 | grad norm= 19.478269577026367
    [ 61/100] L= 0.10472 | grad norm= 115.26640319824219
    [ 66/100] L= 0.11442 | grad norm= 115.26609802246094
    [ 71/100] L= 0.08669 | grad norm= 19.579349517822266
    [ 76/100] L= 0.07696 | grad norm= 19.47224998474121
    [ 81/100] L= 0.08027 | grad norm= 53.744415283203125
    [ 86/100] L= 0.07633 | grad norm= 53.731964111328125
    [ 91/100] L= 0.07649 | grad norm= 19.467098236083984
    [ 96/100] L= 0.07736 | grad norm= 19.476688385009766
    [100/100] L= 0.07590 | grad norm= 19.46072006225586



    
![png](biconvex_parabola_nolens_files/biconvex_parabola_nolens_2_1.png)
    



```python
tlm.show(optics, dim=2, end=60)
tlm.show(optics, dim=3, end=60)
```


<TLMViewer src="./biconvex_parabola_nolens_files/biconvex_parabola_nolens_2.json?url" />



<TLMViewer src="./biconvex_parabola_nolens_files/biconvex_parabola_nolens_3.json?url" />

