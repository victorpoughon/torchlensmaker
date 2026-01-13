# Biconvex Spherical Lens


```python
import torch
import torch.optim as optim
import torchlensmaker as tlm

surface = tlm.Sphere(diameter=15, R=tlm.parameter(25))
lens = tlm.BiLens(surface, material = 'BK7-nd', outer_thickness=1.5)

optics = tlm.Sequential(
    tlm.PointSourceAtInfinity(beam_diameter=18.5),
    tlm.Gap(10),
    lens,
    tlm.Gap(30),
    tlm.FocalPoint(),
)

for name, p in optics.named_parameters():
    print(name, p)

tlm.show(optics, dim=2)
tlm.show(optics, dim=3, sampling={"base": 100})
```

    2.surface1.collision_surface.C Parameter containing:
    tensor(0.0400, dtype=torch.float64, requires_grad=True)



<TLMViewer src="./plot_magnification_files/plot_magnification_0.json?url" />



<TLMViewer src="./plot_magnification_files/plot_magnification_1.json?url" />



```python
tlm.optimize(
    optics,
    optimizer = optim.Adam(optics.parameters(), lr=1e-3),
    sampling = {"base": 10},
    dim = 2,
    num_iter = 100
).plot()

print("Final arc radius:", surface.radius())
print("Outer thickness:", lens.outer_thickness())
print("Inner thickness:", lens.inner_thickness())

tlm.show(optics, dim=2)
tlm.show(optics, dim=3, sampling={"base": 100})
```

    [  1/100] L= 1.56896 | grad norm= 161.9366932906054
    [  6/100] L= 0.78248 | grad norm= 155.03163471742798
    [ 11/100] L= 0.10873 | grad norm= 8.597288480284188
    [ 16/100] L= 0.36217 | grad norm= 146.74322240955695
    [ 21/100] L= 0.20955 | grad norm= 147.74137333401978
    [ 26/100] L= 0.15194 | grad norm= 82.59451332417002
    [ 31/100] L= 0.16661 | grad norm= 82.7744689023683
    [ 36/100] L= 0.11278 | grad norm= 8.862112336619262
    [ 41/100] L= 0.11690 | grad norm= 9.115631464186738
    [ 46/100] L= 0.11255 | grad norm= 8.84745742210896
    [ 51/100] L= 0.10857 | grad norm= 8.586596272281632
    [ 56/100] L= 0.10876 | grad norm= 8.599382275065514
    [ 61/100] L= 0.11234 | grad norm= 8.834141243988748
    [ 66/100] L= 0.11303 | grad norm= 8.877469060183374
    [ 71/100] L= 0.11182 | grad norm= 8.800366538928339
    [ 76/100] L= 0.10944 | grad norm= 8.64474637061142
    [ 81/100] L= 0.10798 | grad norm= 8.546473741008505
    [ 86/100] L= 0.10818 | grad norm= 8.560481822989201
    [ 91/100] L= 0.10933 | grad norm= 8.637634069506536
    [ 96/100] L= 0.10824 | grad norm= 8.564525030391259
    [100/100] L= 0.10813 | grad norm= 8.55690246266337



    
![png](plot_magnification_files/plot_magnification_2_1.png)
    


    Final arc radius: 33.2393986177735
    Outer thickness: tensor(1.5000, dtype=torch.float64, grad_fn=<LinalgVectorNormBackward0>)
    Inner thickness: tensor(3.2144, dtype=torch.float64, grad_fn=<LinalgVectorNormBackward0>)



<TLMViewer src="./plot_magnification_files/plot_magnification_2.json?url" />



<TLMViewer src="./plot_magnification_files/plot_magnification_3.json?url" />



```python
part = tlm.export.lens_to_part(lens)
tlm.show_part(part)
```


<em>part display not supported in vitepress</em>

