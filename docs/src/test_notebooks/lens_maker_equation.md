# Lens Maker Equation (using Refractive Surface)

Here we create a lens with a fixed spherical shape and compute what its focal length should be, using the formula known as the Lens Maker Equation. Then, we use torchlensmaker forward pass to compute the loss and check that it is close to zero, i.e. that the rays correctly converge to the expected focal point.

Note that it will not be exactly zero because the lens maker equation is an approximation (and a pretty bad one).


```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchlensmaker as tlm

import math
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice

# Lens diameter: 40mm
lens_diameter = 40.0

# Circular surface radius
# Using the cartesian sign convention
R1, R2 = 200, -250

# Indices of refraction
N1, N2 = 1.0, 1.5

# Lens Maker Equation
expected_focal_length = 1 / (
    ((N2 - N1) / N1) * (1/R1 - 1/R2)
)

print("Expected focal length (lens's maker equation):", expected_focal_length)

# Surface shapes of the lens
surface1 = tlm.Sphere(lens_diameter, R1)
surface2 = tlm.Sphere(lens_diameter, R2)

# Setup the optical stack with incident parallel rays
optics = nn.Sequential(
    tlm.PointSourceAtInfinity(lens_diameter,  material=tlm.NonDispersiveMaterial(N1)),
    tlm.Gap(10.),

    tlm.RefractiveSurface(surface1, material=tlm.NonDispersiveMaterial(N2), anchors=("origin", "extent")),
    tlm.Gap(10.),
    tlm.RefractiveSurface(surface2, material=tlm.NonDispersiveMaterial(N1), anchors=("extent", "origin")),
    
    tlm.Gap(expected_focal_length - 5.),  # focal length wrt to center of lens
    tlm.FocalPoint(),
)


# Evaluate model with 20 rays
output = optics(tlm.default_input(dim=2, dtype=torch.float64, sampling={"base": 20}))

loss = output.loss.item()
print("Loss:", loss)

# Render
tlm.show(optics, dim=2, end=250)

```

    Expected focal length (lens's maker equation): 222.2222222222222
    Loss: 0.053437498604945864



<TLMViewer src="./lens_maker_equation_tlmviewer/lens_maker_equation_0.json?url" />


# Next

* optimize the gap with VariableGap to improve the best fit focal distance estimation
* resample the shape to a parabola to reduce spherical aberation
