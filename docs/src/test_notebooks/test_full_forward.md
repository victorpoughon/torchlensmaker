```python
import torch
import torch.nn as nn
import torchlensmaker as tlm
import torch.optim as optim

# y = a*x^2
surface = tlm.Parabola(diameter=15, A=tlm.parameter(0.015))

lens = tlm.BiLens(surface, material="BK7", outer_thickness=1.0)

optics = nn.Sequential(
    tlm.PointSourceAtInfinity(beam_diameter=18.5),
    tlm.Wavelength(500, 800),
    tlm.Gap(10),
    lens,
    tlm.Gap(50),
    tlm.FocalPoint(),
)

print(optics)
```

    Sequential(
      (0): PointSourceAtInfinity()
      (1): Wavelength()
      (2): Gap()
      (3): BiLens(
        (surface1): RefractiveSurface(
          (element): Sequential(
            (0): CollisionSurface()
            (1): RefractiveBoundary()
          )
        )
        (gap): Gap()
        (surface2): RefractiveSurface(
          (element): Sequential(
            (0): CollisionSurface()
            (1): RefractiveBoundary()
          )
        )
      )
      (4): Gap()
      (5): FocalPoint()
    )



```python
s1 = tlm.SphereR(diameter=15, R=7.5)

optics = tlm.Sequential(
    tlm.PointSource(beam_angular_size=20),
    tlm.Gap(15),
    tlm.KinematicSurface(nn.Sequential(
        tlm.CollisionSurface(s1),
        tlm.RefractiveBoundary("SF10-nd", "clamp"),
    ), s1, anchors=("origin", "extent")),

    tlm.KinematicSurface(nn.Sequential(
        
        tlm.CollisionSurface(s1),
        tlm.RefractiveBoundary("air", "clamp"),
        
    ), s1, scale=-1, anchors=("extent", "origin")),
)

tlm.show(optics, dim=2, end=20)
```


<TLMViewer src="./test_full_forward_tlmviewer/test_full_forward_0.json?url" />



```python
print(optics)
print()
print(optics[2].element[0])

      
# execute_tree = forward_tree(optics, tlm.default_input(2, torch.float64, sampling={"base": 5}))

# execute_tree[2].element[0].context
```

    Sequential(
      (0): PointSource()
      (1): Gap()
      (2): KinematicSurface(
        (element): Sequential(
          (0): CollisionSurface()
          (1): RefractiveBoundary()
        )
      )
      (3): KinematicSurface(
        (element): Sequential(
          (0): CollisionSurface()
          (1): RefractiveBoundary()
        )
      )
    )
    
    CollisionSurface()



```python
from typing import Any, Iterator
from dataclasses import dataclass

from torchlensmaker.core.full_forward import forward_tree

ins, outs = forward_tree(optics, tlm.default_input(sampling={"base": 5}, dim=2, dtype=torch.float64))
```


```python
outs[optics]
```




    OpticalData(dim=2, dtype=torch.float64, sampling={'base': DenseSampler(size=5)}, transforms=[[IdentityTransform 0x7aa2c803cbc0 dim=2 dtype=torch.float64], [TranslateTransform 0x7aa1b15fb170 dim=2 dtype=torch.float64], [TranslateTransform 0x7aa1b15faed0 dim=2 dtype=torch.float64], [TranslateTransform 0x7aa1b15fb6e0 dim=2 dtype=torch.float64]], P=tensor([[29.8189, -1.6381],
            [29.9484, -0.8783],
            [30.0000,  0.0000],
            [29.9484,  0.8783],
            [29.8189,  1.6381]], dtype=torch.float64), V=tensor([[ 0.9468,  0.3219],
            [ 0.9892,  0.1466],
            [ 1.0000,  0.0000],
            [ 0.9892, -0.1466],
            [ 0.9468, -0.3219]], dtype=torch.float64), normals=None, rays_base=tensor([-0.1745, -0.0873,  0.0000,  0.0873,  0.1745], dtype=torch.float64), rays_object=None, rays_image=None, rays_wavelength=None, var_base=tensor([-0.1745, -0.0873,  0.0000,  0.0873,  0.1745], dtype=torch.float64), var_object=None, var_wavelength=None, material=<torchlensmaker.materials.NonDispersiveMaterial object at 0x7aa1e5ff80b0>, blocked=tensor([False, False, False, False, False]), loss=tensor(0., dtype=torch.float64))


