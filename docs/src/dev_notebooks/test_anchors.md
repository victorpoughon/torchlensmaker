```python
import torchlensmaker as tlm

sphere = tlm.Sphere(10, 15)

optics = tlm.Sequential(
    tlm.RefractiveSurface(sphere, anchors=("origin", "origin"), material="air"),
    tlm.RefractiveSurface(sphere, scale=-1, anchors=("origin", "origin"), material="air")
)

tlm.show2d(optics)
```


<TLMViewer src="./test_anchors_files/test_anchors_0.json?url" />

