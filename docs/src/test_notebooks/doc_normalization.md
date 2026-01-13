## Normalization of coefficients

Most sag functions support normalization, and sag-based surface classes typically expose a `normalize` boolean parameter. When enabled, internal coefficients of the function are stored in normalized form which makes them scale independent. This means that the shape of the modeled surface does not dependent on the diameter of the surface which only changes its scale. To be more precise, the normalized form $\tilde{G}$ of a sag function $G$ is such that:

In 2D:
$$
\tau \, \tilde{g}(\frac{r}{\tau}) = g(r)
$$

In 3D:
$$
\tau \, \tilde{G}(\frac{y}{\tau},\frac{z}{\tau}) = G(y, z)
$$

Where $\tau$ is the normalization radius, typically the surface half-diameter.
In other words, the domain of the sag function whose range represent the surface shape is:
* $[-1 ; 1]$ when normalization is enabled
* $[-\tau ; \tau]$ when normalization is disabled

Normalization is useful typically when coefficients do not represent a physical quantity, like in the aspheric or polynomial models. It is less useful when the coefficient is a radius for example.


```python
import torchlensmaker as tlm
import torch
import json

# Normalized parabola example

# these two surfaces are equivalent
surface = tlm.Parabola(10, A=0.02)
surface2 = tlm.Parabola(10, A=0.1, normalize=True)

optics = tlm.Sequential(
    tlm.ReflectiveSurface(surface),
    tlm.Gap(1),
    tlm.ReflectiveSurface(surface2),
)

tlm.show2d(optics, sampling={}, end=1)
tlm.show3d(optics, sampling={}, end=1)
```


<TLMViewer src="./doc_normalization_files/doc_normalization_0.json?url" />



<TLMViewer src="./doc_normalization_files/doc_normalization_1.json?url" />



```python
import torchlensmaker as tlm
import torch
import json

# Normalized sphere example

# these two surfaces are equivalent
surface = tlm.Sphere(10, R=10)
surface2 = tlm.Sphere(10, R=2, normalize=True)

optics = tlm.Sequential(
    tlm.ReflectiveSurface(surface),
    tlm.Gap(1),
    tlm.ReflectiveSurface(surface2),
)

tlm.show2d(optics, sampling={}, end=1)
tlm.show3d(optics, sampling={}, end=1)
```


<TLMViewer src="./doc_normalization_files/doc_normalization_2.json?url" />



<TLMViewer src="./doc_normalization_files/doc_normalization_3.json?url" />



```python
import torchlensmaker as tlm
import torch
import json

# Normalized asphere example
# these four surfaces are equivalent
asphere1 = tlm.Asphere(10, R=6,  K=0.2, coefficients=[ 1.0000e-02, -1.0000e-04],
                       normalize_conical=False,
                       normalize_aspheric=False)
asphere2 = tlm.Asphere(10, R=6, K=0.2, coefficients=[ 1.2500, -0.3125],
                       normalize_conical=False,
                       normalize_aspheric=True)
asphere3 = tlm.Asphere(10, R=6/5, K=0.2, coefficients=[ 1.0000e-02, -1.0000e-04],
                       normalize_conical=True,
                       normalize_aspheric=False)
asphere4 = tlm.Asphere(10, R=6/5, K=0.2, coefficients=[ 1.2500, -0.3125],
                       normalize_conical=True,
                       normalize_aspheric=True)

optics = tlm.Sequential(
    tlm.ReflectiveSurface(asphere1),
    tlm.Gap(1),
    tlm.ReflectiveSurface(asphere2),
    tlm.Gap(1),
    tlm.ReflectiveSurface(asphere3),
    tlm.Gap(1),
    tlm.ReflectiveSurface(asphere4),
)

tlm.show2d(optics, sampling={}, end=1)
tlm.show3d(optics, sampling={}, end=1)
```


<TLMViewer src="./doc_normalization_files/doc_normalization_4.json?url" />



<TLMViewer src="./doc_normalization_files/doc_normalization_5.json?url" />



```python
import torchlensmaker as tlm
import torch

xy = tlm.XYPolynomial(torch.tensor(
    [[0.1, 0.001],
     [0.1, 0.001]]))


```
