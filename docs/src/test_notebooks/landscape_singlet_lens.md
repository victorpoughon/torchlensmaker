# Landscape Singlet Lens

## Landscape rear configuration


```python
import torch
import torchlensmaker as tlm
import torch.optim as optim

# Setup two spherical surfaces with initial radiuses
surface1 = tlm.Sphere(diameter=30, R=tlm.parameter(-60))
surface2 = tlm.Sphere(diameter=30, R=tlm.parameter(-35))

lens = tlm.Lens(surface1, surface2, material="BK7-nd", outer_thickness=2.2)

focal = 120.5

# Build the optical sequence
optics = tlm.Sequential(
    tlm.ObjectAtInfinity(beam_diameter=8, angular_size=30),
    tlm.Gap(15),
    lens,
    tlm.Gap(focal),
    tlm.ImagePlane(diameter=100, magnification=125.),
)

tlm.show(optics, dim=2, sampling={"base": 10, "object": 5, "sampler": "uniform"})

tlm.plot_magnification(optics, sampling={"base": 10, "object": 5, "sampler": "uniform"})

```


<TLMViewer src="./landscape_singlet_lens_files/landscape_singlet_lens_0.json?url" />



    
![png](landscape_singlet_lens_files/landscape_singlet_lens_2_1.png)
    



```python
# Find the best parameters for the shapes
tlm.optimize(
    optics,
    optimizer = optim.Adam(optics.parameters(), lr=5e-4),
    sampling = {"base": 10, "object": 5, "sampler": "uniform"},
    dim = 2,
    num_iter = 300,
).plot()

# TODO add regularization: both surfaces positive/negative radius

# TODO always zero out y axis of loss plot

tlm.show(optics, dim=2, sampling={"base": 10, "object": 5, "sampler": "uniform"})
tlm.plot_magnification(optics, sampling={"base": 10, "object": 5, "sampler": "uniform"})
```

    [  1/300] L= 18.31260 | grad norm= 9547.974716917457
    [ 16/300] L= 14.16890 | grad norm= 3112.2596999169623
    [ 31/300] L= 12.79578 | grad norm= 1338.9942608022054
    [ 46/300] L= 11.58699 | grad norm= 662.9022929415753
    [ 61/300] L= 10.39884 | grad norm= 520.5845343498661
    [ 76/300] L= 9.20096 | grad norm= 501.9321240313038
    [ 91/300] L= 8.02666 | grad norm= 449.85264745903913
    [106/300] L= 6.92432 | grad norm= 417.2138824894009
    [121/300] L= 5.91195 | grad norm= 381.4307037054442
    [136/300] L= 5.00677 | grad norm= 346.97901281805605
    [151/300] L= 4.21637 | grad norm= 313.2583432557078
    [166/300] L= 3.54194 | grad norm= 279.74931785941556
    [181/300] L= 2.97974 | grad norm= 247.38084091403212
    [196/300] L= 2.52191 | grad norm= 216.56992586562663
    [211/300] L= 2.15775 | grad norm= 187.60055020118077
    [226/300] L= 1.87491 | grad norm= 160.74819827184763
    [241/300] L= 1.66045 | grad norm= 136.2289574967734
    [256/300] L= 1.50170 | grad norm= 114.17641249676032
    [271/300] L= 1.38701 | grad norm= 94.63948996504115
    [286/300] L= 1.30611 | grad norm= 77.58786562350286
    [300/300] L= 1.25349 | grad norm= 63.8284736166912



    
![png](landscape_singlet_lens_files/landscape_singlet_lens_3_1.png)
    



<TLMViewer src="./landscape_singlet_lens_files/landscape_singlet_lens_1.json?url" />



    
![png](landscape_singlet_lens_files/landscape_singlet_lens_3_3.png)
    


## Landscape front configuration


```python
import torch
import torchlensmaker as tlm
import torch.optim as optim

# Setup two spherical surfaces with initial radiuses
surface1 = tlm.Sphere(diameter=30, R=tlm.parameter(torch.tensor(35.)))
surface2 = tlm.Sphere(diameter=30, R=tlm.parameter(torch.tensor(55.)))

lens = tlm.Lens(surface1, surface2, material="BK7-nd", outer_thickness=3)

# Build the optical sequence
optics = tlm.Sequential(
    tlm.ObjectAtInfinity(beam_diameter=30, angular_size=20),
    tlm.Gap(15),
    lens,
    tlm.Gap(20),
    tlm.Aperture(10),
    tlm.Gap(120),
    tlm.ImagePlane(diameter=120, magnification=None),
)

tlm.show(optics, dim=2, sampling={"base": 10, "object": 5, "sampler": "uniform"})
tlm.plot_magnification(optics, sampling={"base": 10, "object": 5, "sampler": "uniform"})
```


<TLMViewer src="./landscape_singlet_lens_files/landscape_singlet_lens_2.json?url" />



    
![png](landscape_singlet_lens_files/landscape_singlet_lens_5_1.png)
    



```python
# Find the best parameters for the shapes
tlm.optimize(
    optics,
    optimizer = optim.Adam(optics.parameters(), lr=1e-3),
    sampling = {"base": 10, "object": 5, "sampler": "uniform"},
    dim = 2,
    num_iter = 400,
).plot()


tlm.show(optics, dim=2, sampling={"base": 10, "object": 11, "sampler": "uniform"})
tlm.plot_magnification(optics, sampling={"base": 10, "object": 5, "sampler": "uniform"})
```

    [  1/400] L= 4.92210 | grad norm= 7978.879761841067
    [ 21/400] L= 0.21049 | grad norm= 640.5633426606289
    [ 41/400] L= 0.20448 | grad norm= 574.470779683938
    [ 61/400] L= 0.18002 | grad norm= 41.5109050602122
    [ 81/400] L= 0.18039 | grad norm= 90.85073024179066
    [101/400] L= 0.17973 | grad norm= 33.47765030333285
    [121/400] L= 0.17950 | grad norm= 2.7223935450377437
    [141/400] L= 0.17934 | grad norm= 5.774101417788389
    [161/400] L= 0.17917 | grad norm= 2.9452922016752705
    [181/400] L= 0.17899 | grad norm= 2.6315281534633868
    [201/400] L= 0.17880 | grad norm= 2.595337090811256
    [221/400] L= 0.17860 | grad norm= 2.583251548751025
    [241/400] L= 0.17839 | grad norm= 2.574671886336434
    [261/400] L= 0.17817 | grad norm= 2.5605743562646173
    [281/400] L= 0.17795 | grad norm= 2.5479982917226973
    [301/400] L= 0.17772 | grad norm= 2.534741465820997
    [321/400] L= 0.17748 | grad norm= 2.5211347694338406
    [341/400] L= 0.17723 | grad norm= 2.5070703595636985
    [361/400] L= 0.17698 | grad norm= 2.4926578023518857
    [381/400] L= 0.17673 | grad norm= 2.4778777427491514
    [400/400] L= 0.17648 | grad norm= 2.4635110325009917



    
![png](landscape_singlet_lens_files/landscape_singlet_lens_6_1.png)
    



<TLMViewer src="./landscape_singlet_lens_files/landscape_singlet_lens_3.json?url" />



    
![png](landscape_singlet_lens_files/landscape_singlet_lens_6_3.png)
    



```python
part = tlm.export.lens_to_part(lens)
tlm.show_part(part)
```


<em>part display not supported in vitepress</em>

