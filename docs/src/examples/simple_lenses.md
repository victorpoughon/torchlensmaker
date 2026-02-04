# Simple lenses

An overview of simple lenses - also called "singlet" lenses. A simple lens is made of two refractive surfaces.

## Biconvex Spherical


```python
import torchlensmaker as tlm


lens = tlm.lenses.symmetric_singlet(
    tlm.Sphere(diameter=10, R=20),
    tlm.OuterGap(0.5),
    material="BK7",
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

lens = tlm.lenses.symmetric_singlet(
    tlm.Parabola(diameter=10, A=0.03),
    tlm.OuterGap(0.5),
    material="BK7",
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

lens = tlm.lenses.symmetric_singlet(
     tlm.Sphere(diameter=10, R=-18),
    tlm.InnerGap(0.5),
    material="BK7",
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

lens = tlm.lenses.singlet(
    tlm.Parabola(diameter=10, A=0.03),
    tlm.OuterGap(0.5),
    tlm.Sphere(diameter=10, R=30),
    material="BK7",
)

optics = tlm.Sequential(
    tlm.PointSourceAtInfinity(beam_diameter=8),
    tlm.Gap(5),
    lens,
)

tlm.show2d(optics, end=20)
```

    [('origin', 'extent'), ('extent', 'origin')]



<TLMViewer src="./simple_lenses_files/simple_lenses_3.json?url" />


## Plano Lens


```python
import torchlensmaker as tlm

lens1 = tlm.lenses.semiplanar_front(
    tlm.Sphere(diameter=10, R=-15),
    tlm.OuterGap(0.8),
    material="BK7",
)

lens2 = tlm.lenses.semiplanar_rear(
    tlm.Sphere(diameter=10, R=-15),
    tlm.InnerGap(0.6),
    material="BK7",
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

