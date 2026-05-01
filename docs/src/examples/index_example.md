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

    [  1/100] L= 519.20715 | grad norm= 108750.8359
    [  6/100] L= 286.02924 | grad norm= 79397.8984
    [ 11/100] L= 129.28835 | grad norm= 52595.5312
    [ 16/100] L= 41.32748 | grad norm= 29305.0977
    [ 21/100] L= 5.80536 | grad norm= 10576.0674
    [ 26/100] L= 0.74943 | grad norm= 2749.7698
    [ 31/100] L= 5.75126 | grad norm= 10428.0840
    [ 36/100] L= 8.78169 | grad norm= 13027.3066
    [ 41/100] L= 7.26108 | grad norm= 11795.9160
    [ 46/100] L= 3.78660 | grad norm= 8312.9033
    [ 51/100] L= 1.21555 | grad norm= 4127.5664
    [ 56/100] L= 0.38931 | grad norm= 474.3787
    [ 61/100] L= 0.55703 | grad norm= 1911.8381
    [ 66/100] L= 0.77395 | grad norm= 2845.7163
    [ 71/100] L= 0.70640 | grad norm= 2591.1572
    [ 76/100] L= 0.51255 | grad norm= 1656.7959
    [ 81/100] L= 0.39441 | grad norm= 573.7497
    [ 86/100] L= 0.38180 | grad norm= 267.8958
    [ 91/100] L= 0.40166 | grad norm= 690.6813
    [ 96/100] L= 0.40321 | grad norm= 713.4008
    [100/100] L= 0.39245 | grad norm= 538.0247



<TLMViewer src="./index_example_files/index_example_0.json?url" />

