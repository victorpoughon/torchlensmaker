```python
import torchlensmaker as tlm
import torch

lens = tlm.lenses.cemented(
        surfaces=[
            tlm.Sphere(diameter=30, R=55),
            tlm.Sphere(diameter=30, R=-55),
            tlm.Sphere(diameter=30, R=-120),
        ],
        gaps=[
            tlm.InnerGap(5.5),
            tlm.InnerGap(0.5),
        ],
        materials=["BK7", "SF10", "air"],
    )

print(lens)
print(tlm.lens_inner_thickness(lens))
print(tlm.lens_outer_thickness(lens))

# Add a light source to test the lens
optics = tlm.Sequential(
    tlm.ObjectAtInfinity(beam_diameter=20, angular_size=0.5, wavelength=(400, 800)),
    tlm.Gap(1),
    lens,
)

tlm.show2d(optics, end=30)
```

    [('origin', 'origin'), ('origin', 'origin'), ('origin', 'origin')]
    Lens(
      (sequence): Sequential(
        (0): RefractiveSurface(
          (collision_surface): CollisionSurface()
          (material): CauchyMaterial()
        )
        (1): Gap()
        (2): RefractiveSurface(
          (collision_surface): CollisionSurface()
          (material): CauchyMaterial()
        )
        (3): Gap()
        (4): RefractiveSurface(
          (collision_surface): CollisionSurface()
          (material): NonDispersiveMaterial(n=1.0002700090408325)
        )
      )
    )
    tensor(6.)
    tensor(2.9738)



<TLMViewer src="./new_lenses_files/new_lenses_0.json?url" />

