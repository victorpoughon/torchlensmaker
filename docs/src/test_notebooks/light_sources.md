# Light sources


```python
import torch
import torch.nn as nn
import torchlensmaker as tlm
import torch.optim as optim

# Idea: Renderwing of light sources: render a bit into the negative t with different color?
# + render the source 'outline' as a line/disk/surface
```

## RaySource

`RaySource()` is a light source that emits a single ray of light. It does not need any sampling information.


```python
optics = tlm.Sequential(
    tlm.Turn([20, 0]),
    tlm.RaySource(material="air")
)

tlm.show(optics, dim=2, end=40, sampling={})
tlm.show(optics, dim=3, end=40, sampling={})
```


<TLMViewer src="./light_sources_files/light_sources_0.json?url" />



<TLMViewer src="./light_sources_files/light_sources_1.json?url" />


## PointSourceAtInfinity

`PointSourceAtInfinity()` represents a point light source "at infinity", meaning that the source is so far away that the rays it emits are perfecly parallel. The number of rays depends on the "base" sampling dimension along the source's beam diameter.


```python
optics = nn.Sequential(
    tlm.Gap(10),
    tlm.Rotate(
        tlm.PointSourceAtInfinity(beam_diameter=18.5),
        angles = (-15, -5),
    ),
    tlm.Gap(10),
    tlm.PointSourceAtInfinity(beam_diameter=18.5),
)

tlm.show2d(optics, end=40, sampling={"base": 30})
tlm.show3d(optics, end=40, sampling={"base": 50})
```


<TLMViewer src="./light_sources_files/light_sources_2.json?url" />



<TLMViewer src="./light_sources_files/light_sources_3.json?url" />


## PointSource

A point source that's positioned in physical space by the kinematic chain. Rays are all emitted from the point source position and are sampled along the "base" dimension, within the domain defined by the beam angular size.


```python
optics = nn.Sequential(
    tlm.Gap(-10),
    tlm.Rotate(
        tlm.PointSource(10),
        [15, 0])
)

tlm.show(optics, dim=2, end=30, sampling={"base": 10, "sampler": "random"})
tlm.show(optics, dim=3, end=100, sampling={"base": 100, "sampler": "random"})
```


<TLMViewer src="./light_sources_files/light_sources_4.json?url" />



<TLMViewer src="./light_sources_files/light_sources_5.json?url" />


## ObjectAtInfinity

An object that's so far away that all light rays coming from the same position on the object are perfectly parallel. Emits light rays along both "base" and "object" sampling dimensions, within the domain defined by the beam diameter and the object angular size.


```python
optics = nn.Sequential(
    tlm.ObjectAtInfinity(beam_diameter=10, angular_size=25),
    tlm.Gap(20),
    tlm.BiLens(tlm.Parabola(diameter=15, A=tlm.parameter(0.02)), material = 'BK7-nd', outer_thickness=1.0),
)

tlm.show2d(optics, end=50)
tlm.show3d(optics, end=200, sampling={"base": 20, "object": 20})
```


<TLMViewer src="./light_sources_files/light_sources_6.json?url" />



<TLMViewer src="./light_sources_files/light_sources_7.json?url" />


## Object

An object that's positioned in physical space by the kinematic chain. Emits light rays along both "base" and "object" sampling dimensions, within the domain defined by the object diameter and the beam angular size.


```python
surface = tlm.Parabola(diameter=15, A=tlm.parameter(0.02))
lens = tlm.BiLens(surface, material = 'BK7-nd', outer_thickness=1.0)

object_distance = 50

optics = nn.Sequential(
    tlm.Gap(-object_distance),
    tlm.Object(beam_angular_size=5, object_diameter=5),
    tlm.Gap(object_distance),
    tlm.Gap(20),
    lens,
)

tlm.show(optics, dim=2, end=200)
tlm.show(optics, dim=3, end=200, sampling={"base": 10, "object": 10})
```


<TLMViewer src="./light_sources_files/light_sources_8.json?url" />



<TLMViewer src="./light_sources_files/light_sources_9.json?url" />


## Wavelength

Adds a wavelength variable to existing rays. Duplicates existing light rays for each sampled wavelength value. Values are samples along the "wavelength" dimension, within the bounds defined by the object min and max arguments.


```python
optics = nn.Sequential(
    tlm.Gap(-1),
    tlm.PointSourceAtInfinity(beam_diameter=12),
    tlm.Gap(1),
    tlm.Wavelength(400, 800),
    tlm.BiLens(tlm.Parabola(diameter=15, A=tlm.parameter(0.02)), material = 'SF10', outer_thickness=1.0),
)

# As with other dimensions, configure the sampled wavelengths with the sampling dictionary
tlm.show2d(optics, end=5, sampling={"base": 10, "wavelength": 10})
tlm.show3d(optics, end=5, sampling={"base": 15, "wavelength": [400, 600, 600]})
```


<TLMViewer src="./light_sources_files/light_sources_10.json?url" />



<TLMViewer src="./light_sources_files/light_sources_11.json?url" />

