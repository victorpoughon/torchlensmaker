# Implicit Disk

A disk surface, in 2D or 3D, represented by an implicit function.

## Code example

```python
import torchlensmaker as tlm

surface = tlm.ImplicitDisk(diameter=20)

optics = tlm.Sequential(tlm.ReflectiveSurface(surface))
tlm.show2d(optics)
tlm.show3d(optics)
```

## Parameters

| Name       | Trainable | Description          |
| ---------- | --------- | -------------------- |
| `diameter` | Yes       | Diameter of the disk |


## Constructor

## Raytracing

## Math Model

A disk of radius $R$ in the plane perpendicular to the principal axis.

### In 2D

In 2D the model reduces to a line segment from $[0, -R]$ to $[0, +R]$. The implicit function formula is split into two cases.

* If $|r| \leq R$:

Implicit function:

$$F(x,r) = |x|$$

Gradient:

$$
\begin{align}
\nabla_x F(x,r) &= \text{sgn}(x) \\
\nabla_r F(x,r) &= 0
\end{align}
$$

Hessian is zero.

* If $|r| \gt R$:

Implicit function:

$$
F(x, r) = 
 \sqrt{(|r| - R)^2 + x^2}
$$

To simplify notation, let $s=(|r|-R)^2 + x^2$.

The implicit function simplifies to:

$$
F(x, r) = 
 \sqrt{s}
$$

Gradient:

$$
\begin{align}
\nabla_x F(x, r) &= \frac{x}{\sqrt{s}} \\
\nabla_r F(x,r) &= \frac{\text{sgn}(r)(|r| - R)}{\sqrt{s}}
\end{align}
$$

Hessian:

$$
\begin{align}
\nabla_{xx} F(x,r) &= \frac{s - x^2}{s\sqrt{s}} \\
\nabla_{xr} F(x,r) &= \frac{-x \cdot \text{sgn}(r)(|r| - R)}{s\sqrt{s}} = \nabla_{rx} F(x,r) \\
\nabla_{rr} F(x,r) &= \frac{s - (|r| - R)^2}{s\sqrt{s}}\\
\end{align}
$$


### In 3D

The disk is rotationally symmetric around the X axis so we have: $r = \sqrt{y^2 + z^2}$.
And again to simplify notation, let $s=(r-R)^2 + x^2$.

The implicit function formula is split into two cases:

* If $r \leq R$:

TODO


* If $r \gt R$:

Implicit function:

$$
\begin{align}
F(x, y, z) &= \sqrt{s} \\
&= \sqrt{(r - R)^2 + x^2} \\
&= \sqrt{(\sqrt{y^2+z^2} - R)^2 + x^2}
\end{align}
$$

Gradient:

$$
\begin{align}
\nabla_x F(x,y,z) &= \frac{x}{\sqrt{s}} \\
\nabla_y F(x,y,z) &= \frac{y(r-R)}{r\sqrt{s}} \\
\nabla_z F(x,y,z) &= \frac{z(r-R)}{r\sqrt{s}}
\end{align}
$$

Hessian:

$$
\begin{align}
\nabla_{xx} F(x,y,z) &= \frac{s - x^2}{s \sqrt{s}}\\
\nabla_{yy} F(x,y,z) &= \frac{(r-R)r^2s + y^2(Rs - r(r-R)^2)}{r^3 s \sqrt{s}}\\
\nabla_{zz} F(x,y,z) &= \frac{(r-R)r^2s + z^2(Rs - r(r-R)^2)}{r^3 s \sqrt{s}}\\
\nabla_{xy} F(x,y,z) &= \frac{-xy(r -R)}{rs\sqrt{s}} \\
\nabla_{xz} F(x,y,z) &= \frac{-xz(r -R)}{rs\sqrt{s}} \\
\nabla_{yz} F(x,y,z) &=  \frac{yz [Rs - r(r-R)^2]}{r^3 s \sqrt{s}} \\
\end{align}
$$
