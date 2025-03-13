# Reflecting Telescope

A reflecting telescope, in its basic form, is a very simple design. Two surfaces both reflect light to focus it at a single point.

![Newton reflecting Telescope](./newton_reflecting_telescope.jpeg)

*Image from Wikipedia*

For this example, our telescope will be made of two convace mirrors. To spice things up, we'll say that the primary mirror is parabolic, and the secondary is spherical. Of course this can easily be changed, so feel free to download this notebook and play with it. In this example, we will keep the position of the two mirrors constant, and try to optimize the two mirrors curvatures jointly.


```python
import torchlensmaker as tlm

primary = tlm.Parabola(35., A=tlm.parameter(-0.0001))
secondary = tlm.Sphere(35., R=tlm.parameter(450.0))

optics = tlm.Sequential(
    tlm.Gap(-100),
    tlm.PointSourceAtInfinity(beam_diameter=30),
    tlm.Gap(100),
    
    tlm.ReflectiveSurface(primary),
    tlm.Gap(-80),

    tlm.ReflectiveSurface(secondary),

    tlm.Gap(100),
    tlm.FocalPoint(),
)

tlm.show2d(optics)
tlm.show3d(optics)
```


<TLMViewer src="./reflecting_telescope_files/reflecting_telescope_0.json?url" />



<TLMViewer src="./reflecting_telescope_files/reflecting_telescope_1.json?url" />


Now, as you can see light isn't being focused at all. We have wrapped both surfaces arguments in `tlm.parameter()`. Internally, this creates a `nn.Parameter()` so that PyTorch can optimize them. Let's run a standard Adam optimizer for 100 iterations, with 10 rays samples.


```python
import torch.optim as optim

tlm.optimize(
    optics,
    optimizer = optim.Adam(optics.parameters(), lr=3e-4),
    sampling = {"base": 10},
    dim = 2,
    num_iter = 100
).plot()
```

    [  1/100] L=  4.143 | grad norm= 5082.363918620529
    [  6/100] L=  2.142 | grad norm= 4340.7287936145685
    [ 11/100] L=  0.756 | grad norm= 4644.1259709838105
    [ 16/100] L=  0.759 | grad norm= 4603.423573570586
    [ 21/100] L=  1.231 | grad norm= 4325.102890947149
    [ 26/100] L=  0.651 | grad norm= 4534.0182712760015
    [ 31/100] L=  0.422 | grad norm= 4376.47656474204
    [ 36/100] L=  0.136 | grad norm= 4394.467760993692
    [ 41/100] L=  0.376 | grad norm= 4442.9357634169955
    [ 46/100] L=  0.229 | grad norm= 4411.379267391225
    [ 51/100] L=  0.499 | grad norm= 4307.708008600472
    [ 56/100] L=  0.664 | grad norm= 4447.6069665453715
    [ 61/100] L=  0.342 | grad norm= 4308.933241431921
    [ 66/100] L=  0.037 | grad norm= 4349.463429603535
    [ 71/100] L=  0.197 | grad norm= 4363.575517844985
    [ 76/100] L=  0.145 | grad norm= 4351.460623838482
    [ 81/100] L=  0.120 | grad norm= 4312.586251795746
    [ 86/100] L=  0.246 | grad norm= 4291.967671380488
    [ 91/100] L=  0.213 | grad norm= 4292.15663167869
    [ 96/100] L=  0.004 | grad norm= 1206.8889840433037
    [100/100] L=  0.011 | grad norm= 4312.224145720371



    
![png](reflecting_telescope_files/reflecting_telescope_4_1.png)
    



```python
tlm.show(optics, dim=2)
tlm.show(optics, dim=3)
```


<TLMViewer src="./reflecting_telescope_files/reflecting_telescope_2.json?url" />



<TLMViewer src="./reflecting_telescope_files/reflecting_telescope_3.json?url" />

