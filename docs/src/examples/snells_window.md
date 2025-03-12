# Snell's Window


```python
import torchlensmaker as tlm

optics = tlm.Sequential(
    tlm.PointSource(beam_angular_size=105, material="water-nd"),
    tlm.Gap(2),
    tlm.RefractiveSurface(tlm.CircularPlane(15), critical_angle="reflect", material="air"),
)

# material: str for simple dispersion models
# material: MaterialModel for advanced use

tlm.show(optics, dim=2, end=2, sampling={"base": 100})
tlm.show(optics, dim=3, end=2, sampling={"base": 2000})
```


<TLMViewer src="./snells_window_files/snells_window_0.json?url" />



<TLMViewer src="./snells_window_files/snells_window_1.json?url" />



```python
import torch
output = optics(tlm.default_input(dim=2, dtype=torch.float64, sampling={"base": 10}))

print(output.material)
```

    NonDispersiveMaterial(name='air', n=1.00027)

