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

tlm.simple_optimize(optics, tlm.optim.Adam(optics.parameters(), lr=5e-4), 100)

tlm.show2d(optics, title="Landscape Lens")
```

    [  1/100] L= 519.20642 | grad norm= 108750.7266
    [  6/100] L= 286.02881 | grad norm= 79397.8359
    [ 11/100] L= 129.28790 | grad norm= 52595.4297
    [ 16/100] L= 41.32740 | grad norm= 29305.0684
    [ 21/100] L= 5.80529 | grad norm= 10576.0078
    [ 26/100] L= 0.74945 | grad norm= 2749.8591
    [ 31/100] L= 5.75129 | grad norm= 10428.1104
    [ 36/100] L= 8.78164 | grad norm= 13027.2666
    [ 41/100] L= 7.26105 | grad norm= 11795.8838
    [ 46/100] L= 3.78657 | grad norm= 8312.8740
    [ 51/100] L= 1.21552 | grad norm= 4127.4795
    [ 56/100] L= 0.38931 | grad norm= 474.2907
    [ 61/100] L= 0.55704 | grad norm= 1911.8560
    [ 66/100] L= 0.77394 | grad norm= 2845.6758
    [ 71/100] L= 0.70640 | grad norm= 2591.1355
    [ 76/100] L= 0.51256 | grad norm= 1656.8081
    [ 81/100] L= 0.39441 | grad norm= 573.7310
    [ 86/100] L= 0.38180 | grad norm= 267.9317
    [ 91/100] L= 0.40166 | grad norm= 690.6925
    [ 96/100] L= 0.40321 | grad norm= 713.4318
    [100/100] L= 0.39246 | grad norm= 537.9373



<TLMViewer src="./index_example_files/index_example_0.json?url" />

