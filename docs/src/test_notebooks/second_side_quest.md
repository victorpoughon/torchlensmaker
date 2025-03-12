# Second side quest


```python
import torch
import torch.nn as nn
import torchlensmaker as tlm

s1 = tlm.Sphere(diameter=15, R=12)


class Id(nn.Module):
    def forward(self, x):
        return x


# checks at each element boundary
# surface normals are unit
# shapes are correct
# wavelengths are in expected range

class Check(nn.Module):
    def __init__(self, tag):
        super().__init__()
        self.tag = tag

    def forward(self, inputs):
        print(f"[{self.tag}]", "P", inputs.P.shape)
        print(f"[{self.tag}]", "normals", inputs.normals.shape if inputs.normals is not None else None)
        return inputs


optics = tlm.Sequential(
    tlm.PointSource(beam_angular_size=20),
    tlm.Gap(15),
    tlm.KinematicSurface(nn.Sequential(
        
        tlm.CollisionSurface(s1),
        tlm.RefractiveBoundary("SF10-nd", "clamp"),
    ), s1, anchors=("origin", "extent")),
    Check("a"),
    tlm.KinematicSurface(nn.Sequential(
        
        tlm.CollisionSurface(s1),
        tlm.RefractiveBoundary("air", "clamp"),
        
    ), s1, scale=-1, anchors=("extent", "origin")),
)

tlm.show(optics, dim=2, end=20)
```

    [a] P torch.Size([10, 2])
    [a] normals None



<TLMViewer src="./second_side_quest_tlmviewer/second_side_quest_0.json?url" />



```python
surface = tlm.Sphere(diameter=15, R=-18)

optics = nn.Sequential(
    tlm.Offset(
        tlm.Rotate(
            tlm.RaySource(material="SF10-nd"),
            [-20, 0]),
        y=10),
    tlm.Gap(10),
    
    tlm.KinematicSurface(nn.Sequential(
        
        tlm.CollisionSurface(s1),
        tlm.RefractiveBoundary("air", "clamp"),
        
    ), s1, scale=-1, anchors=("extent", "origin")),
    
    tlm.Gap(30),
    tlm.FocalPoint(),
)


tlm.show(optics, dim=2, sampling={"base": 10})
```


<TLMViewer src="./second_side_quest_tlmviewer/second_side_quest_1.json?url" />



```python
surface = tlm.Sphere(diameter=15, R=-18)

optics = nn.Sequential(
    tlm.Offset(
        tlm.Rotate(
            tlm.RaySource(material="SF10-nd"),
            [-20, 0]),
        y=10),
    tlm.Gap(10),

    #tlm.RefractiveSurface(s1, "air", -1, ("extent", "origin"), "drop"),
    tlm.ReflectiveSurface(s1, -1, ("extent", "origin")),
    
    tlm.Gap(30),
    tlm.FocalPoint(),
)


tlm.show(optics, dim=2, sampling={"base": 10})
```


<TLMViewer src="./second_side_quest_tlmviewer/second_side_quest_2.json?url" />



```python
surface = tlm.Sphere(diameter=15, R=-18)

optics = nn.Sequential(
    tlm.Offset(
        tlm.Rotate(
            tlm.RaySource(material="SF10-nd"),
            [-20, 0]),
        y=10),
    tlm.Gap(10),

    tlm.Offset(
        tlm.ReflectiveSurface(s1, -1, ("extent", "origin")),
        y=10),
    
    tlm.Gap(30),
    tlm.FocalPoint(),
)

print(optics)

tlm.show(optics, dim=2, sampling={"base": 10})
```

    Sequential(
      (0): Offset(
        (element): Rotate(
          (element): RaySource()
        )
      )
      (1): Gap()
      (2): Offset(
        (element): ReflectiveSurface(
          (element): Sequential(
            (0): CollisionSurface()
            (1): ReflectiveBoundary()
          )
        )
      )
      (3): Gap()
      (4): FocalPoint()
    )



<TLMViewer src="./second_side_quest_tlmviewer/second_side_quest_3.json?url" />

