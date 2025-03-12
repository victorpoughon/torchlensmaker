# Asphere

The class `tlm.Asphere` implements the typical [Aphere model](https://en.m.wikipedia.org/wiki/Aspheric_lens), with a few changes:
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

## Axially symmetric 3D asphere

In the 3D rotationally symmetric case, we have $r^2 = y^2 + z^2$.

The derivative with respect to y (or z, by symmetry) is:

$$
F'_y(x,y,z) = \frac{C y}{\sqrt{1-(1+K) (y^2+z^2) C^2}} + y \Big( 4 \alpha_4 (y^2 + z^2) + 6 \alpha_6 (y^2 + z^2)^2 + \ldots \Big)
$$




```python
import torchlensmaker as tlm
from torchlensmaker.testing.basic_transform import basic_transform

scene = tlm.viewer.new_scene("3D")

test_data = [
    (basic_transform(1.0, "origin", [0., 0., 0.], [0., 0., 0.]), tlm.Asphere(diameter=30, R=-15, K=-1.6, A4=0.00012)),
    #(basic_transform(1.0, "origin", [0., 0., 0.], [5., 0., 0.]), tlm.Sphere(diameter=30, R=30)),
]

test_surfaces = [s for t, s in test_data]
test_transforms = [t for t, s in test_data]


def demo():
    realized_transforms = [t(s) for t, s in zip(test_transforms, test_surfaces)]
    scene = tlm.viewer.new_scene("3D")
    scene["data"].append(tlm.viewer.render_surfaces(test_surfaces, realized_transforms, dim=3))
    tlm.viewer.display_scene(scene)

demo()
```


<TLMViewer src="./asphere_tlmviewer/asphere_0.json?url" />

