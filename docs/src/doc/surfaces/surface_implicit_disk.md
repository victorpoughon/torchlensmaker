# Implicit Disk

A disk surface, in 2D or 3D, represented by an implicit function.

## Code example

```python
import torchlensmaker as tlm

surface = tlm.ImplicitDisk(diameter=20)

optics = tlm.Sequential(tlm.ReflectiveSurface(surface))
tlm.show2d(optics)
tlm.show3d(optics)
```

## Parameters

| Name       | Trainable | Description          |
| ---------- | --------- | -------------------- |
| `diameter` | Yes       | Diameter of the disk |


## Constructor



## Raytracing
