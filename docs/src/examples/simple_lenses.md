# Simple lenses

An overview of simple lenses - also called "singlet" lenses. A simple lens is made of two refractive surfaces.

## Biconvex Spherical


```python
import torchlensmaker as tlm

lens = tlm.BiLens(
    tlm.Sphere(diameter=10, R=20), material="BK7-nd", outer_thickness=0.5
)

optics = tlm.Sequential(
    tlm.PointSourceAtInfinity(beam_diameter=8),
    tlm.Gap(5),
    lens,
)

tlm.show2d(optics, end=20)
```


<TLMViewer src="./simple_lenses_files/simple_lenses_0.json?url" />


## Biconvex Parabolic


```python
import torchlensmaker as tlm

lens = tlm.BiLens(
    tlm.Parabola(diameter=10, A=0.03), material="BK7-nd", outer_thickness=0.5
)

optics = tlm.Sequential(
    tlm.PointSourceAtInfinity(beam_diameter=8),
    tlm.Gap(5),
    lens,
)

tlm.show2d(optics, end=20)
```


<TLMViewer src="./simple_lenses_files/simple_lenses_1.json?url" />


## Biconcave spherical (diverging lens)


```python
import torchlensmaker as tlm

# The shape given to BiLens is the first surface.
# The second surface is mirrored by its Y axis.
# Hence to make a diverging lens, r is negative here
# Note we also use inner_thickness to specify the lens thickness
# because the inner thickness is smallest in a diverging lens.
lens = tlm.BiLens(
    tlm.Sphere(diameter=10, R=-18), material="BK7-nd", inner_thickness=0.5
)

optics = tlm.Sequential(
    tlm.PointSourceAtInfinity(beam_diameter=8),
    tlm.Gap(5),
    lens,
)

tlm.show2d(optics, end=20)
```


<TLMViewer src="./simple_lenses_files/simple_lenses_2.json?url" />


## Meniscus Lens


```python
import torchlensmaker as tlm

lens = tlm.Lens(
    tlm.Parabola(diameter=10, A=0.03),
    tlm.Sphere(diameter=10, R=30),
    material="BK7-nd",
    outer_thickness=0.5,
)

optics = tlm.Sequential(
    tlm.PointSourceAtInfinity(beam_diameter=8),
    tlm.Gap(5),
    lens,
)

tlm.show2d(optics, end=20)
```


<TLMViewer src="./simple_lenses_files/simple_lenses_3.json?url" />


## Plano Lens

To make a plano-concave or plano-convex lens, use the `tlm.PlanoLens` class.


```python
import torchlensmaker as tlm

lens1 = tlm.PlanoLens(
    tlm.Sphere(diameter=10, R=-15),
    material="BK7-nd",
    outer_thickness=0.8,
)

lens2 = tlm.PlanoLens(
    tlm.Sphere(diameter=10, R=15),
    material="BK7-nd",
    inner_thickness=0.6,
    reverse=True,
)

optics = tlm.Sequential(
    tlm.PointSourceAtInfinity(beam_diameter=8),
    tlm.Gap(5),
    lens1,
    tlm.Gap(5),
    lens2,
)

tlm.show2d(optics, end=10)
```


<TLMViewer src="./simple_lenses_files/simple_lenses_4.json?url" />

