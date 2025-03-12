# Lenses


```python
import torch.nn as nn
import torchlensmaker as tlm

gap = tlm.Gap(5)

optics = nn.Sequential(
    # 'Bilens' for mirrored symmetric lenses - biconvex / biconcave
    tlm.BiLens(tlm.Sphere(diameter=10, R=20), material = 'BK7-nd', outer_thickness=0.0),
    gap,

    tlm.BiLens(tlm.Sphere(diameter=10, R=20), material = 'BK7-nd', inner_thickness=2.5),
    gap,

    tlm.BiLens(tlm.Sphere(diameter=10, R=-20), material = 'BK7-nd', outer_thickness=2.5),
    gap,
    
    tlm.BiLens(tlm.Sphere(diameter=10, R=-20), material = 'BK7-nd', inner_thickness=0.1),
    gap,

    # 'Lens' for general purpose asymmetric lenses
    tlm.Lens(tlm.Sphere(diameter=10, R=30), tlm.Parabola(diameter=10, A=-0.05), material = 'BK7-nd', outer_thickness=0.5),
    gap,

    tlm.Lens(tlm.Sphere(diameter=10, R=-30), tlm.Parabola(diameter=10, A=-0.02), material = 'BK7-nd', outer_thickness=0.5),
    gap,

    # 'PlanoLens' for semi planar lenses - plano-convex, etc.
    tlm.PlanoLens(tlm.Sphere(diameter=10, R=-30), material = 'BK7-nd', outer_thickness=0),
    gap,

    tlm.PlanoLens(tlm.Sphere(diameter=10, R=30), material = 'BK7-nd', inner_thickness=0.2),
    gap,

    # note reverse=True swap the two surface, and flips them
    tlm.PlanoLens(tlm.Sphere(diameter=10, R=30), material = 'BK7-nd', inner_thickness=0.2, reverse=True),
    gap,
)    


tlm.show(optics, dim=3)
```


<TLMViewer src="./lenses_tlmviewer/lenses_0.json" />



```python
for element in optics:
    if isinstance(element, tlm.LensBase):
        part = tlm.export.lens_to_part(element)
        tlm.show_part(part)
```


<em>part display not supported in vitepress</em>



<em>part display not supported in vitepress</em>



<em>part display not supported in vitepress</em>



<em>part display not supported in vitepress</em>



<em>part display not supported in vitepress</em>



<em>part display not supported in vitepress</em>



<em>part display not supported in vitepress</em>



<em>part display not supported in vitepress</em>



<em>part display not supported in vitepress</em>



```python
# Export all lenses to step files
# tlm.export.export_all_step(optics, "./")
```
