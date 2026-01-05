# Surfaces

Torch Lens Maker is designed to support many surfaces. Currently supported are:

* [CircularPlane](/doc/modeling/surfaces#circularplane): A circular planar surface (aka a disk)
* [Sphere](/doc/modeling/surfaces#sphere): with curvature parameterization
* [SphereR](/doc/modeling/surfaces#spherer): Sphere with radius parameterization
* [Parabola](/doc/modeling/surfaces#parabola): A parabola on the X axis $x = a r^2$
* [Asphere](/doc/modeling/surfaces#asphere): The [asphere model](https://en.wikipedia.org/wiki/Aspheric_lens) often used in optics

::: warning Work in progress
More surface types coming soon, hopefully 😁 I also want to document how to add custom surfaces easily, as a lot of work as gone into that while designing the library. Basically any sag function $x = g(r)$ can be added, or even any implicit surface described by $F(x,y,z) = 0$, not necessarily axially symmetric.
:::

 ## CircularPlane




```python
import torchlensmaker as tlm

surface = tlm.CircularPlane(diameter=4.0)
optics = tlm.Sequential(
    tlm.PointSource(10),
    tlm.Gap(5),
    tlm.Turn((45, 0)),
    tlm.ReflectiveSurface(surface),
)

tlm.show2d(optics, end=5)
```


<TLMViewer src="./surfaces_files/surfaces_0.json?url" />


## Sphere

A section of a sphere, parameterized by signed curvature.
Curvature is the inverse of radius: C = 1/R.

This parameterization is useful because it enables clean representation of
an infinite radius section of sphere (which is really a plane), and also
enables changing the sign of C during optimization.

In 2D, this surface is an arc of circle.
In 3D, this surface is a section of a sphere (wikipedia calls it a "spherical cap")

For high curvature arcs (close to a half circle), it's better to use the
SphereR class which uses radius parameterization and polar distance
functions. In fact this class cannot represent an exact half circle (R =
D/2) due to the gradient becoming infinite, use SphereR instead.


```python
import torchlensmaker as tlm

surface = tlm.Sphere(diameter=10, R=-25)

optics = tlm.Sequential(
    tlm.PointSourceAtInfinity(beam_diameter=5),
    tlm.Gap(5),
    tlm.Turn((20, 0)),
    tlm.ReflectiveSurface(surface),
)

tlm.show2d(optics, end=12)
tlm.show3d(optics, end=12)
```


<TLMViewer src="./surfaces_files/surfaces_1.json?url" />



<TLMViewer src="./surfaces_files/surfaces_2.json?url" />


## SphereR

A section of a sphere, parameterized by signed radius.

This parameterization is useful to represent high curvature sections
including a complete half-sphere. However it's poorly suited to represent
low curvature sections that are closer to a planar surface.

In 2D, this surface is an arc of circle.
In 3D, this surface is a section of a sphere (wikipedia call it a "spherical cap")


```python
import torchlensmaker as tlm

surface1 = tlm.SphereR(diameter=10, R=5)
surface2 = tlm.SphereR(diameter=10, R=-5)

optics = tlm.Sequential(
    tlm.ReflectiveSurface(surface1),
    tlm.Gap(10),
    tlm.ReflectiveSurface(surface2),
)

tlm.show2d(optics, end=12)
tlm.show3d(optics, end=12)
```


<TLMViewer src="./surfaces_files/surfaces_3.json?url" />



<TLMViewer src="./surfaces_files/surfaces_4.json?url" />


## Parabola

A parabolic surface on the X axis: $X = A R^2$. 


```python
import torchlensmaker as tlm

surface = tlm.Parabola(diameter=10, A=0-.03)

optics = tlm.Sequential(
    tlm.PointSource(35),
    tlm.Gap(-1/(4*surface.A.item())),
    tlm.ReflectiveSurface(surface),
)

tlm.show2d(optics, end=4)
tlm.show3d(optics, end=4)
```


<TLMViewer src="./surfaces_files/surfaces_5.json?url" />



<TLMViewer src="./surfaces_files/surfaces_6.json?url" />


## Asphere

The typical [Aphere model](https://en.m.wikipedia.org/wiki/Aspheric_lens), with a few changes:
* X is the principal optical axis in torchlensmaker
* Internally, the radius parameter R is represented as the curvature $C = \frac{1}{R}$, to allow changing the sign of curvature during optimization
* k is the conic constant

$$
X(r) = \frac{C r^2}{1+\sqrt{1-(1+K)r^2 C^2}} + \alpha_4 r^4 + \alpha_6 r^6 \ldots
$$

In 2D, the derivative with respect to r is:

$$
\nabla_r X(r) = \frac{C r}{\sqrt{1-(1+K)r^2 C^2}} + 4 \alpha_4 r^3 + 6 \alpha_6 r^5 \ldots
$$

In the 3D rotationally symmetric case, we have $r^2 = y^2 + z^2$.

The derivative with respect to y (or z, by symmetry) is:

$$
F'_y(x,y,z) = \frac{C y}{\sqrt{1-(1+K) (y^2+z^2) C^2}} + y \Big( 4 \alpha_4 (y^2 + z^2) + 6 \alpha_6 (y^2 + z^2)^2 + \ldots \Big)
$$



```python
import torchlensmaker as tlm


surface = tlm.Asphere(diameter=30, R=-15, K=-1.6, A4=0.00012)

optics = tlm.Sequential(
    tlm.PointSourceAtInfinity(25),
    tlm.Gap(5),
    tlm.RefractiveSurface(surface, material="water-nd"),
)

tlm.show2d(optics, end=10)
tlm.show3d(optics, end=10)
```


<TLMViewer src="./surfaces_files/surfaces_7.json?url" />



<TLMViewer src="./surfaces_files/surfaces_8.json?url" />

