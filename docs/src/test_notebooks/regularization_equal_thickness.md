# Regularization - Equal Thickness Lenses

In this example, we design a two lens system where the first lens is plano-convex, and the second lens is biconvex symmetric. Two parameters are used to describe the curvature of each lens.

This problem has many solutions because different surface shape combinations can achieve the desired focal length. Using regularization, we add the additional constraint that the inner thickness of each lens should be equal. This leads to a unique solution.


```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchlensmaker as tlm
import math


lens_diamater = 30
focal_length = 45

lens_outer_thickness = 1.0

# Shape of the curved surface of the plano convex lens
surface_convex = tlm.Parabola(lens_diamater, tlm.parameter(-0.005))

# Shape of the two curved surfaces of the biconvex symmetric lens
surface_biconvex = tlm.Parabola(lens_diamater, tlm.parameter(0.005))

# Convex-planar lens
lens_plano = tlm.PlanoLens(
    surface_convex,
    material = "BK7-nd",
    outer_thickness = lens_outer_thickness,
    reverse=True,
)

# Biconvex lens
lens_biconvex = tlm.BiLens(
    surface_biconvex,
    material = "air",
    outer_thickness = lens_outer_thickness,
)

optics = nn.Sequential(
    tlm.PointSourceAtInfinity(0.9*lens_diamater),
    tlm.Gap(10.),
    lens_biconvex,
    tlm.Gap(3.),
    lens_plano,
    tlm.Gap(focal_length),
    tlm.FocalPoint(),
)

tlm.show(optics, dim=2)
tlm.show(optics, dim=3)
```


<TLMViewer src="./regularization_equal_thickness_tlmviewer/regularization_equal_thickness_0.json?url" />



<TLMViewer src="./regularization_equal_thickness_tlmviewer/regularization_equal_thickness_1.json?url" />



```python
# The regularization function
# This adds a term to the loss function to ensure
# both lenses' inner thicknesses are equal
def regu_equalthickness(optics):
    t0 = lens_plano.inner_thickness()
    t1 = lens_biconvex.inner_thickness()
    return 100*torch.pow(t0 - t1, 2)


tlm.optimize(
    optics,
    optimizer = optim.Adam(optics.parameters(), lr=3e-4),
    sampling = {"base": 10},
    dim = 2,
    num_iter = 100
).plot()


def print_thickness(lens_name, lens):
    # TODO thickness at a specific radial distance
    print(f"{lens_name: <25} inner: {lens.inner_thickness().item():.3f} outer: {lens.outer_thickness().item():.3f}")

print_thickness("Plano-convex", lens_plano)
print_thickness("Bi-convex", lens_biconvex)
```

    [  1/100] L=  5.693 | grad norm= 365.67909387157397
    [  6/100] L=  5.144 | grad norm= 367.51879541852355
    [ 11/100] L=  4.593 | grad norm= 368.97986098990674
    [ 16/100] L=  4.039 | grad norm= 370.0669665807215
    [ 21/100] L=  3.484 | grad norm= 370.7870990977098
    [ 26/100] L=  2.928 | grad norm= 371.14965463656665
    [ 31/100] L=  2.372 | grad norm= 371.1664493575745
    [ 36/100] L=  1.816 | grad norm= 370.8516177922875
    [ 41/100] L=  1.262 | grad norm= 370.221398449639
    [ 46/100] L=  0.709 | grad norm= 369.29382617789076
    [ 51/100] L=  0.181 | grad norm= 240.85435844478266
    [ 56/100] L=  0.264 | grad norm= 366.9804612572807
    [ 61/100] L=  0.322 | grad norm= 366.81655117431734
    [ 66/100] L=  0.172 | grad norm= 119.2313414589781
    [ 71/100] L=  0.147 | grad norm= 86.85149720705407
    [ 76/100] L=  0.166 | grad norm= 240.7654251537622
    [ 81/100] L=  0.147 | grad norm= 86.84679008058838
    [ 86/100] L=  0.136 | grad norm= 118.9199452550004
    [ 91/100] L=  0.137 | grad norm= 118.92601686222727
    [ 96/100] L=  0.135 | grad norm= 86.61108524379534
    [100/100] L=  0.130 | grad norm= 118.86034388580518



    
![png](regularization_equal_thickness_files/regularization_equal_thickness_3_1.png)
    


    Plano-convex              inner: 5.609 outer: 1.000
    Bi-convex                 inner: 12.304 outer: 1.000



```python
tlm.show_part(tlm.export.lens_to_part(lens_plano))
tlm.show_part(tlm.export.lens_to_part(lens_biconvex))
```


<em>part display not supported in vitepress</em>



<em>part display not supported in vitepress</em>

