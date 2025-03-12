# Export with build123


```python
import torchlensmaker as tlm
from torchlensmaker.export_build123d import surface_to_sketch


# TODO add SphereR

test_shapes = [
    tlm.Sphere(15.0, 20.0),
    tlm.Sphere(15.0, -20.0),
    tlm.Parabola(15.0, 0.02),
    tlm.Parabola(15.0, -0.02),
    tlm.CircularPlane(15.0),
]

for shape in test_shapes:
    sk = surface_to_sketch(shape)
    if not tlm.viewer.vue_format_requested():
        display(sk)
```
