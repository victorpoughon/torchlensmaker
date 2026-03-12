# Light sources

Currently supported light sources:

* [RaySource](/modeling/light_sources#raysource): A single light ray. (Sampling dimensions: `wavel`)
* [PointSource](/modeling/light_sources#pointsource): A point source. (Sampling dimensions: `pupil, wavel`)
* [PointSourceAtInfinity](/modeling/light_sources#pointsourceatinfinity): A point source at infinity. (Sampling dimensions: `pupil, wavel`)
* [Object](/modeling/light_sources#object): A circular object on the kinematic chain. (Sampling dimensions: `pupil, field, wavel`)
* [ObjectAtInfinity](/modeling/light_sources#objectatinfinity): A circular object at infinity. (Sampling dimensions: `pupil, field, wavel`)


## RaySource

A light source that emits a single ray of light. The only sampling dimension is wavelength (`wavel`).


```python
import torchlensmaker as tlm

optics = tlm.Sequential(tlm.Rotate([20, 0]), tlm.RaySource(wavelength=[500, 800]))

optics.set_sampling2d(wavel=10)

tlm.show2d(optics, end=40)
tlm.show3d(optics, end=40)
```


<TLMViewer src="./light_sources_files/light_sources_0.json?url" />



<TLMViewer src="./light_sources_files/light_sources_1.json?url" />


## PointSourceAtInfinity

A point light source "at infinity", meaning that the source is so far away that the rays it emits are perfecly parallel. The number of rays depends on the `pupil` sampling dimension along the source's beam diameter. The element's position on the kinematic chain represents the start point of the rays.


```python
optics = tlm.Sequential(
    tlm.Gap(10),
    tlm.SubChain(
        tlm.Rotate((-15, -5)),
        tlm.PointSourceAtInfinity(beam_diameter=18.5),
    ),
    tlm.Gap(10),
)

optics.set_sampling2d(pupil=30)
tlm.show2d(optics, end=40)

optics.set_sampling3d(pupil=200)
tlm.show3d(optics, end=40)
```


<TLMViewer src="./light_sources_files/light_sources_2.json?url" />



<TLMViewer src="./light_sources_files/light_sources_3.json?url" />


## PointSource

A point source that's positioned in physical space by the kinematic chain. Rays are all emitted from the point source position and are sampled along the `pupil` dimension, within the domain defined by the beam angular size.


```python
optics = tlm.Sequential(
    tlm.Gap(-10),
    tlm.Rotate((15, 0)),
    tlm.PointSource(10)
)

tlm.show2d(optics, end=30)
tlm.show3d(optics, end=100)
```


<TLMViewer src="./light_sources_files/light_sources_4.json?url" />



<TLMViewer src="./light_sources_files/light_sources_5.json?url" />


## ObjectAtInfinity

An object that's so far away that all light rays coming from the same position on the object are perfectly parallel. Emits light rays along both `pupil` and `field` sampling dimensions, within the domain defined by the beam diameter and the object angular size. The position of this optical element on the kinematic chain represents the start point of the rays.


```python
optics = tlm.Sequential(
    tlm.ObjectAtInfinity(beam_diameter=10, angular_size=25),
    tlm.Gap(20),
    tlm.lenses.symmetric_singlet(
        tlm.Parabola(diameter=15, A=0.02),
        tlm.OuterGap(1.0),
        material="BK7",
    ),
)

tlm.show2d(optics, end=50)
tlm.show3d(optics, end=50)
```


<TLMViewer src="./light_sources_files/light_sources_6.json?url" />



<TLMViewer src="./light_sources_files/light_sources_7.json?url" />


## Object

An object that's positioned in physical space by the kinematic chain. Emits light rays along both `pupil` and `field` sampling dimensions, within the domain defined by the object diameter and the beam angular size.


```python
surface = tlm.Parabola(diameter=15, A=tlm.parameter(0.02))
lens = tlm.lenses.symmetric_singlet(surface, tlm.OuterGap(1.0), material="BK7")

object_distance = 50

optics = tlm.Sequential(
    tlm.Gap(-object_distance),
    tlm.Object(beam_angular_size=5, object_diameter=5),
    tlm.Gap(object_distance),
    tlm.Gap(20),
    lens,
)

tlm.show2d(optics, end=60)
tlm.show3d(optics, end=60)
```


<TLMViewer src="./light_sources_files/light_sources_8.json?url" />



<TLMViewer src="./light_sources_files/light_sources_9.json?url" />

