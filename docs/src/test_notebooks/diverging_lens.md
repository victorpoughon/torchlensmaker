# Biconcave diverging lens


```python
import torch
import torch.nn as nn
import torchlensmaker as tlm
import torch.optim as optim

surface = tlm.Parabola(20., A=tlm.parameter(-0.03))

lens = tlm.BiLens(surface, material = 'BK7-nd', inner_thickness=1.0)

optics = nn.Sequential(
    tlm.PointSourceAtInfinity(15),
    tlm.Gap(10), 
    lens,
    tlm.Gap(-25),
    tlm.FocalPoint(),
)

tlm.show(optics, dim=2)
tlm.show(optics, dim=3)
```


<TLMViewer src="./diverging_lens_tlmviewer/diverging_lens_0.json" />



<TLMViewer src="./diverging_lens_tlmviewer/diverging_lens_1.json" />



```python
# Perform optimization in 2D
tlm.optimize(
    optics,
    optimizer = optim.Adam(optics.parameters(), lr=1e-3),
    sampling = {"base": 10},
    dim = 2,
    num_iter = 100
).plot()

print("Outer thickness:", lens.outer_thickness().item())
print("Inner thickness:", lens.inner_thickness().item())

# Render again after optimization
tlm.show(optics, dim=2)
tlm.show(optics, dim=3)
```

    [  1/100] L=  3.569 | grad norm= 369.0638887895675
    [  6/100] L=  1.910 | grad norm= 298.1817418957014
    [ 11/100] L=  0.532 | grad norm= 264.0099844682011
    [ 16/100] L=  0.488 | grad norm= 246.5403605685403
    [ 21/100] L=  0.579 | grad norm= 245.21615426150962
    [ 26/100] L=  0.158 | grad norm= 51.71364142807621
    [ 31/100] L=  0.304 | grad norm= 242.1971960610418
    [ 36/100] L=  0.166 | grad norm= 91.474397096122
    [ 41/100] L=  0.197 | grad norm= 250.9868779711139
    [ 46/100] L=  0.142 | grad norm= 51.29803917204158
    [ 51/100] L=  0.165 | grad norm= 91.45295261426011
    [ 56/100] L=  0.139 | grad norm= 51.21930330858593
    [ 61/100] L=  0.153 | grad norm= 51.601133882237164
    [ 66/100] L=  0.135 | grad norm= 51.136728733088056
    [ 71/100] L=  0.134 | grad norm= 90.24171392308608
    [ 76/100] L=  0.143 | grad norm= 51.346448576140574
    [ 81/100] L=  0.144 | grad norm= 90.61273979705645
    [ 86/100] L=  0.139 | grad norm= 51.232710499591796
    [ 91/100] L=  0.137 | grad norm= 90.37706146667213
    [ 96/100] L=  0.137 | grad norm= 51.167725107252
    [100/100] L=  0.133 | grad norm= 90.1925674729305



    
![png](diverging_lens_files/diverging_lens_2_1.png)
    


    Outer thickness: 4.610795108462309
    Inner thickness: 1.0



<TLMViewer src="./diverging_lens_tlmviewer/diverging_lens_2.json" />



<TLMViewer src="./diverging_lens_tlmviewer/diverging_lens_3.json" />



```python
part = tlm.export.lens_to_part(lens)
tlm.show_part(part)
```


<em>part display not supported in vitepress</em>

