# Reflecting Telescope

A reflecting telescope, in its basic form, is a very simple design. Two surfaces both reflect light to focus it at a single point.

![Newton reflecting Telescope](./newton_reflecting_telescope.jpeg)

*Image from Wikipedia*

For this example, our telescope will be made of two convave mirrors. To spice things up, we'll say that the primary mirror is parabolic, and the secondary is spherical. Of course this can easily be changed, so feel free to download this notebook and play with it. In this example, we will keep the position of the two mirrors constant, and try to optimize the two mirrors curvatures jointly.


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

optics.set_sampling2d(pupil=10)

tlm.optimize(
    optics,
    optimizer = optim.Adam(optics.parameters(), lr=3e-4),
    dim = 2,
    num_iter = 40
).plot()
```

    [  1/40] L= 4.14344 | grad norm= 5082.36376953125
    [  3/40] L= 0.47920 | grad norm= 4675.3154296875
    [  5/40] L= 1.93830 | grad norm= 4376.82177734375
    [  7/40] L= 1.99583 | grad norm= 4347.521484375
    [  9/40] L= 0.98014 | grad norm= 4451.8486328125
    [ 11/40] L= 0.75573 | grad norm= 4644.12646484375
    [ 13/40] L= 1.55097 | grad norm= 4721.04931640625
    [ 15/40] L= 1.22367 | grad norm= 4666.35107421875
    [ 17/40] L= 0.14839 | grad norm= 4522.39599609375
    [ 19/40] L= 1.01303 | grad norm= 4365.19921875
    [ 21/40] L= 1.23147 | grad norm= 4325.10302734375
    [ 23/40] L= 0.75589 | grad norm= 4373.916015625
    [ 25/40] L= 0.29608 | grad norm= 4495.216796875
    [ 27/40] L= 0.78485 | grad norm= 4545.55078125
    [ 29/40] L= 0.48030 | grad norm= 4498.51025390625
    [ 31/40] L= 0.42176 | grad norm= 4376.47705078125
    [ 33/40] L= 0.80629 | grad norm= 4319.34619140625
    [ 35/40] L= 0.50553 | grad norm= 4350.7177734375
    [ 37/40] L= 0.36812 | grad norm= 4455.01806640625
    [ 39/40] L= 0.73781 | grad norm= 4494.76953125
    [ 40/40] L= 0.63783 | grad norm= 4479.029296875



    
![png](reflecting_telescope_files/reflecting_telescope_4_1.png)
    



```python
tlm.show(optics, dim=2)
tlm.show(optics, dim=3)
```


<TLMViewer src="./reflecting_telescope_files/reflecting_telescope_2.json?url" />



<TLMViewer src="./reflecting_telescope_files/reflecting_telescope_3.json?url" />

