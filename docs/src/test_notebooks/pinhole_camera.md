# Pinhole camera


```python
import torchlensmaker as tlm

optics = tlm.Sequential(
    tlm.ObjectAtInfinity(beam_diameter=12, angular_size=40),
    tlm.Gap(25),
    tlm.Aperture(diameter=.5),
    tlm.Gap(40),
    tlm.ImagePlane(diameter=50),
)

sampling = {"base": 50, "object": 15}
tlm.show(optics, dim=2, sampling=sampling)
tlm.plot_magnification(optics, sampling=sampling)
```


<TLMViewer src="./pinhole_camera_files/pinhole_camera_0.json?url" />



    
![png](pinhole_camera_files/pinhole_camera_1_1.png)
    

