# Reflecting Telescope

A reflecting telescope, in its basic form, is a very simple design. Two surfaces both reflect light to focus it at a single point.

![Newton reflecting Telescope](./newton_reflecting_telescope.jpeg)

*Image from Wikipedia*

For this example, our telescope will be made of two convave mirrors. To spice things up, we'll say that the primary mirror is parabolic, and the secondary is spherical. Of course this can easily be changed, so feel free to download this notebook and play with it. In this example, we will keep the position of the two mirrors constant, and try to optimize the two mirrors curvatures jointly.


```python
import torchlensmaker as tlm

primary = tlm.Parabola(35., A=-0.0001, trainable=True)
secondary = tlm.SphereByCurvature(35., C=1/450.0, trainable=True)

source = tlm.Sequential(
    tlm.Gap(-100),
    tlm.PointSourceAtInfinity(beam_diameter=30),
)

model = tlm.Sequential(
    tlm.Gap(100),
    
    tlm.ReflectiveSurface(primary),
    tlm.Gap(-80),

    tlm.ReflectiveSurface(secondary),

    tlm.Gap(100),
)

target = tlm.FocalPoint()

optics = tlm.Sequential(source, model, target)

tlm.show2d(optics)
tlm.show3d(optics)
```


<TLMViewer src="./reflecting_telescope_files/reflecting_telescope_0.json?url" />



<TLMViewer src="./reflecting_telescope_files/reflecting_telescope_1.json?url" />


Now, as you can see light isn't being focused at all. We have wrapped both surfaces arguments in `tlm.parameter()`. Internally, this creates a `nn.Parameter()` so that PyTorch can optimize them. Let's run a standard Adam optimizer for 100 iterations, with 10 rays samples.


```python
# Optimization in 3D

import torch.optim as optim

optics.set_sampling3d(pupil=60)

root = tlm.SequentialData.empty(dim=3)
inputs = source.sequential(root)

tlm.optimize(
    model,
    inputs,
    target,
    optimizer = optim.Adam(optics.parameters(), lr=3e-5),
    num_iter = 150
).plot()
```

    [  1/150] L= 4.74863 | grad norm= 5825.4546
    [  9/150] L= 3.01573 | grad norm= 5638.7612
    [ 17/150] L= 1.36351 | grad norm= 5454.4297
    [ 25/150] L= 0.16155 | grad norm= 5279.1870
    [ 33/150] L= 0.36405 | grad norm= 5252.1646
    [ 41/150] L= 0.20423 | grad norm= 5314.7461
    [ 49/150] L= 0.11227 | grad norm= 5276.8345
    [ 57/150] L= 0.08746 | grad norm= 5299.0830
    [ 65/150] L= 0.00774 | grad norm= 5289.3267
    [ 73/150] L= 0.03391 | grad norm= 5292.0522
    [ 81/150] L= 0.01360 | grad norm= 5289.4688
    [ 89/150] L= 0.02353 | grad norm= 5285.0127
    [ 97/150] L= 0.00521 | grad norm= 5288.2783
    [105/150] L= 0.01189 | grad norm= 5288.9897
    [113/150] L= 0.01476 | grad norm= 5289.2764
    [121/150] L= 0.00564 | grad norm= 5288.1670
    [129/150] L= 0.00787 | grad norm= 5286.5474
    [137/150] L= 0.02941 | grad norm= 5290.8628
    [145/150] L= 0.00410 | grad norm= 3212.4607
    [150/150] L= 0.04207 | grad norm= 5282.2925



    
![png](reflecting_telescope_files/reflecting_telescope_4_1.png)
    



```python
# Optimization in 2D

import torch.optim as optim

optics.set_sampling2d(pupil=10)

root = tlm.SequentialData.empty(dim=2)
inputs = source.sequential(root)

tlm.optimize(
    model,
    inputs,
    target,
    optimizer = optim.Adam(optics.parameters(), lr=3e-6),
    num_iter = 150
).plot()
```

    [  1/150] L= 0.02946 | grad norm= 4609.6958
    [  9/150] L= 0.00551 | grad norm= 3135.7239
    [ 17/150] L= 0.00466 | grad norm= 3135.8296
    [ 25/150] L= 0.00349 | grad norm= 1293.1007
    [ 33/150] L= 0.00375 | grad norm= 1293.1350
    [ 41/150] L= 0.00337 | grad norm= 1290.3088
    [ 49/150] L= 0.00383 | grad norm= 1293.1542
    [ 57/150] L= 0.00352 | grad norm= 1293.1254
    [ 65/150] L= 0.00383 | grad norm= 1290.2700
    [ 73/150] L= 0.00342 | grad norm= 1290.3282
    [ 81/150] L= 0.00371 | grad norm= 1293.1667
    [ 89/150] L= 0.00422 | grad norm= 1293.2322
    [ 97/150] L= 0.00332 | grad norm= 1290.3610
    [105/150] L= 0.00393 | grad norm= 1293.2131
    [113/150] L= 0.00350 | grad norm= 1293.1718
    [121/150] L= 0.00399 | grad norm= 1290.3020
    [129/150] L= 0.00365 | grad norm= 1290.3499
    [137/150] L= 0.00395 | grad norm= 1290.3221
    [145/150] L= 0.00352 | grad norm= 1290.3828
    [150/150] L= 0.00350 | grad norm= 1290.3893



    
![png](reflecting_telescope_files/reflecting_telescope_5_1.png)
    



```python
tlm.show(optics, dim=2)
tlm.show(optics, dim=3)
```


<TLMViewer src="./reflecting_telescope_files/reflecting_telescope_2.json?url" />



<TLMViewer src="./reflecting_telescope_files/reflecting_telescope_3.json?url" />

