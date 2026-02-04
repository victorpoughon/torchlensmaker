# Index example

This is the example script for the documentation welcome page.


```python
import torchlensmaker as tlm

optics = tlm.Sequential(
    tlm.ObjectAtInfinity(beam_diameter=10, angular_size=20),
    tlm.Gap(15),
    tlm.RefractiveSurface(tlm.Sphere(diameter=25, R=-45.0), material="BK7"),
    tlm.Gap(3),
    tlm.RefractiveSurface(tlm.Sphere(diameter=25, R=tlm.parameter(-20)), material="air"),
    tlm.Gap(100),
    tlm.ImagePlane(50),
)

tlm.optimize(optics, tlm.optim.Adam(optics.parameters(), lr=5e-4), 100)

tlm.show2d(optics, title="Landscape Lens")
```

    [  1/100] L= 519.20715 | grad norm= 108925.8515625
    [  6/100] L= 286.02957 | grad norm= 79517.96875
    [ 11/100] L= 129.29071 | grad norm= 52670.93359375
    [ 16/100] L= 41.33147 | grad norm= 29346.001953125
    [ 21/100] L= 5.80771 | grad norm= 10591.7138671875
    [ 26/100] L= 0.74863 | grad norm= 2750.785400390625
    [ 31/100] L= 5.74839 | grad norm= 10438.837890625
    [ 36/100] L= 8.77968 | grad norm= 13042.439453125
    [ 41/100] L= 7.26133 | grad norm= 11811.314453125
    [ 46/100] L= 3.78813 | grad norm= 8325.71484375
    [ 51/100] L= 1.21672 | grad norm= 4136.1806640625
    [ 56/100] L= 0.38944 | grad norm= 478.22735595703125
    [ 61/100] L= 0.55667 | grad norm= 1911.8594970703125
    [ 66/100] L= 0.77379 | grad norm= 2848.22607421875
    [ 71/100] L= 0.70658 | grad norm= 2594.679931640625
    [ 76/100] L= 0.51283 | grad norm= 1660.11767578125
    [ 81/100] L= 0.39453 | grad norm= 575.999755859375
    [ 86/100] L= 0.38175 | grad norm= 267.0213623046875
    [ 91/100] L= 0.40157 | grad norm= 690.9361572265625
    [ 96/100] L= 0.40319 | grad norm= 714.3287963867188
    [100/100] L= 0.39244 | grad norm= 539.0952758789062



<TLMViewer src="./index_example_files/index_example_0.json?url" />



```python
import numpy as np

# Spot diaggram at 0, 5 and 10 and 15 degrees incidence angles
"""
{"base":1000, "object": [
    [np.deg2rad(0), 0.],
    [np.deg2rad(5), 0.],
    [np.deg2rad(10), 0.]]
}
"""

# TODO fix spot diagram
# _ = tlm.spot_diagram(optics, sampling, col="object", figsize=(12, 12))
```




    '\n{"base":1000, "object": [\n    [np.deg2rad(0), 0.],\n    [np.deg2rad(5), 0.],\n    [np.deg2rad(10), 0.]]\n}\n'


