# Index example

This is the example script for the documentation welcome page.


```python
import torchlensmaker as tlm

optics = tlm.Sequential(
    tlm.ObjectAtInfinity(beam_diameter=10, angular_size=20),
    tlm.Gap(15),
    tlm.RefractiveSurface(
        tlm.SphereByCurvature(diameter=25, C=1 / -45.0), materials=("air", "BK7")
    ),
    tlm.Gap(3),
    tlm.RefractiveSurface(
        tlm.SphereByCurvature(diameter=25, C=tlm.parameter(1 / -20)),
        materials=("BK7", "air"),
    ),
    tlm.Gap(100),
    tlm.ImagePlane(50),
)

tlm.optimize(optics, tlm.optim.Adam(optics.parameters(), lr=5e-4), 100)

tlm.show2d(optics, title="Landscape Lens")
```

    [  1/100] L= 519.20715 | grad norm= 144701.9062
    [  6/100] L= 140.03506 | grad norm= 74219.1406
    [ 11/100] L= 5.55948 | grad norm= 13893.8457
    [ 16/100] L= 17.59479 | grad norm= 25605.5820
    [ 21/100] L= 37.07623 | grad norm= 37472.4727
    [ 26/100] L= 20.06305 | grad norm= 27396.8965
    [ 31/100] L= 2.29991 | grad norm= 8115.5757
    [ 36/100] L= 2.25732 | grad norm= 8033.8696
    [ 41/100] L= 5.48624 | grad norm= 13793.1152
    [ 46/100] L= 2.96627 | grad norm= 9598.9111
    [ 51/100] L= 0.64508 | grad norm= 1380.6758
    [ 56/100] L= 1.12212 | grad norm= 4511.6348
    [ 61/100] L= 1.30558 | grad norm= 5240.3320
    [ 66/100] L= 0.72177 | grad norm= 2208.6897
    [ 71/100] L= 0.63016 | grad norm= 1164.9724
    [ 76/100] L= 0.74223 | grad norm= 2391.6133
    [ 81/100] L= 0.64401 | grad norm= 1382.4163
    [ 86/100] L= 0.59675 | grad norm= 276.4730
    [ 91/100] L= 0.62374 | grad norm= 1061.0367
    [ 96/100] L= 0.60668 | grad norm= 689.2250
    [100/100] L= 0.59424 | grad norm= 61.5149



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


