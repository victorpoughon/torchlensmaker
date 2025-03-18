# Index example

This is the example script for the documentation welcome page.


```python
import torchlensmaker as tlm

# Define the optical sequence: a simple lens made of two refractive surfaces
optics = tlm.Sequential(
    tlm.ObjectAtInfinity(beam_diameter=10, angular_size=20),
    tlm.Gap(15),
    tlm.RefractiveSurface(tlm.Sphere(diameter=25, R=-45.0), material="BK7-nd"),
    tlm.Gap(3),
    tlm.RefractiveSurface(tlm.Sphere(diameter=25, R=tlm.parameter(-20)), material="air"),
    tlm.Gap(100),
    tlm.ImagePlane(50),
)

# Optimize the radius of curvature
tlm.optimize(optics, tlm.optim.Adam(optics.parameters(), lr=5e-4), {"base": 10, "object": 5}, 100)

# Show 2D layout
tlm.show2d(optics, title="Landscape Lens")
```

    [  1/100] L= 161.110 | grad norm= 34027.92132918352
    [  6/100] L= 88.273 | grad norm= 24824.72953856118
    [ 11/100] L= 39.426 | grad norm= 16373.459257966973
    [ 16/100] L= 12.248 | grad norm= 9001.44816291341
    [ 21/100] L=  1.564 | grad norm= 3071.495725565856
    [ 26/100] L=  0.324 | grad norm= 1120.9160184216457
    [ 31/100] L=  2.004 | grad norm= 3485.6156820326296
    [ 36/100] L=  2.871 | grad norm= 4211.980082166615
    [ 41/100] L=  2.265 | grad norm= 3719.6828440763443
    [ 46/100] L=  1.120 | grad norm= 2535.24844357612
    [ 51/100] L=  0.341 | grad norm= 1167.9203867628376
    [ 56/100] L=  0.132 | grad norm= 13.884880785644617
    [ 61/100] L=  0.207 | grad norm= 703.0796822820761
    [ 66/100] L=  0.267 | grad norm= 942.9408673391762
    [ 71/100] L=  0.232 | grad norm= 809.7331350515325
    [ 76/100] L=  0.167 | grad norm= 479.4960834122824
    [ 81/100] L=  0.134 | grad norm= 126.23700435637802
    [ 86/100] L=  0.134 | grad norm= 129.77923653501958
    [ 91/100] L=  0.141 | grad norm= 241.5122542562588
    [ 96/100] L=  0.139 | grad norm= 225.6937244646029
    [100/100] L=  0.135 | grad norm= 156.7784315553501



<TLMViewer src="./index_example_files/index_example_0.json?url" />



```python
import numpy as np

# Spot diaggram at 0, 5 and 10 and 15 degrees incidence angles
sampling = {"base":1000, "object": [
    [np.deg2rad(0), 0.],
    [np.deg2rad(5), 0.],
    [np.deg2rad(10), 0.]]
}

_ = tlm.spot_diagram(optics, sampling, col="object", figsize=(12, 12))
```


    
![png](index_example_files/index_example_2_0.png)
    

