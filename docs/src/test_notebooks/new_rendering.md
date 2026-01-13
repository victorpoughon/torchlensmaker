```python
import torch
import torchlensmaker as tlm

doublet = tlm.Sequential(
    tlm.RefractiveSurface(tlm.Sphere(4.0, C=0.135327), material=tlm.NonDispersiveMaterial(1.517)),
    tlm.Gap(1.05),
    tlm.RefractiveSurface(tlm.Sphere(4.0, C=-0.19311), material=tlm.NonDispersiveMaterial(1.649)),
    tlm.Gap(0.4),
    tlm.RefractiveSurface(tlm.Sphere(4.0, C=-0.06164), material="air"),
)

optics = tlm.Sequential(
    tlm.PointSourceAtInfinity(3.0),
    tlm.Gap(2),
    doublet,
)

print(optics)
```

    Sequential(
      (0): PointSourceAtInfinity()
      (1): Gap()
      (2): Sequential(
        (0): RefractiveSurface(
          (collision_surface): CollisionSurface()
        )
        (1): Gap()
        (2): RefractiveSurface(
          (collision_surface): CollisionSurface()
        )
        (3): Gap()
        (4): RefractiveSurface(
          (collision_surface): CollisionSurface()
        )
      )
    )



```python
scene = tlm.viewer.new_scene("2D")

# solution1: modules define their artist, rendering still happens in the artist object
# solution2: move artist code to module

tlm.viewer.display_scene(scene)
```


<TLMViewer src="./new_rendering_files/new_rendering_0.json?url" />

