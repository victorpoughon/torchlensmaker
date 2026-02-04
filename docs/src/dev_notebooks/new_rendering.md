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
      (0): PointSourceAtInfinity(
        (module_2d): PointSourceAtInfinity2D(
          (sampler_pupil): LinspaceSampler1D(N=5)
          (sampler_field): ZeroSampler1D()
          (sampler_wavelength): LinspaceSampler1D(N=5)
          (material): NonDispersiveMaterial(n=1.0002700090408325)
          (geometry): ObjectAtInfinityGeometry2D()
        )
        (module_3d): PointSourceAtInfinity3D(
          (sampler_pupil): DiskSampler2D(Nrho=5, Ntheta=5)
          (sampler_field): ZeroSampler2D()
          (sampler_wavelength): LinspaceSampler1D(N=5)
          (material): NonDispersiveMaterial(n=1.0002700090408325)
          (geometry): ObjectAtInfinityGeometry3D()
        )
      )
      (1): Gap()
      (2): Sequential(
        (0): RefractiveSurface(
          (collision_surface): CollisionSurface()
          (material): NonDispersiveMaterial(n=1.5169999599456787)
        )
        (1): Gap()
        (2): RefractiveSurface(
          (collision_surface): CollisionSurface()
          (material): NonDispersiveMaterial(n=1.6490000486373901)
        )
        (3): Gap()
        (4): RefractiveSurface(
          (collision_surface): CollisionSurface()
          (material): NonDispersiveMaterial(n=1.0002700090408325)
        )
      )
    )



```python
scene = tlm.new_scene("2D")

# solution1: modules define their artist, rendering still happens in the artist object
# solution2: move artist code to module

tlm.display_scene(scene)
```


<TLMViewer src="./new_rendering_files/new_rendering_0.json?url" />

