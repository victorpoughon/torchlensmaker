# Cooke triplet

Let's try to model a Cooke Triplet, working from this screenshot from an old patent, taken from Wikipedia:

![image.png](cooke_triplet_files/f54c02f5-674c-4b1c-a455-9ef50d350bc9.png)


First, let's define the surfaces as spherical, using the radius of curvatures provided. Lenses diameters are not given, I'm gonna guess 30 and 25 to start with:


```python
import torch
import torchlensmaker as tlm

d1, d2 = 30, 25

r1 = tlm.Sphere(d1, 26.4)
r2 = tlm.Sphere(d1, -150.7)
r3 = tlm.Sphere(d2, -29.8)
r4 = tlm.Sphere(d2, 24.2)
r5 = tlm.Sphere(d1, 150.7)
r6 = tlm.Sphere(d1, -26.4)
```

Next, we can define the material models for the lenses. In a Cooke Triplet, two materials are used, with the center one being denser to refract light more.


```python
material1 = tlm.NonDispersiveMaterial(1.5108)
material2 = tlm.NonDispersiveMaterial(1.6042)

L1 = tlm.Lens(r1, r2, material=material1, inner_thickness=5.9)
L2 = tlm.Lens(r3, r4, material=material2, inner_thickness=0.2)
L3 = tlm.Lens(r5, r6, material=material1, inner_thickness=5.9)
```

The distance between the last lens and the image plane is unclear from the document. Let's guess an inital value of 85, but wrap it in `tlm.parameter()`, so we can optimize it later.


```python
focal = tlm.parameter(85)
```

Finally, we can define our optical sequence model, by stacking everying together in a `tlm.Sequential` object.


```python
optics = tlm.Sequential(
    tlm.ObjectAtInfinity(15, 25),
    L1,
    tlm.Gap(10.9),
    L2,
    tlm.Gap(3.1),
    tlm.Aperture(18),
    tlm.Gap(9.4),
    L3,
    tlm.Gap(focal),
    tlm.ImagePlane(65),
)
```

And we can view the system in 2D with `tlm.show2d()`. Note that if don't specify sampling parameters, it will try to use sensible ones by default.


```python
tlm.show2d(optics)
```


<TLMViewer src="./cooke_triplet_files/cooke_triplet_0.json?url" />



```python
sampling = {"base":1000, "object": 4}
tlm.show3d(optics, sampling)
f, _ = tlm.spot_diagram(optics, sampling=sampling, row="object", figsize=(12, 12))
```


<TLMViewer src="./cooke_triplet_files/cooke_triplet_1.json?url" />



    
![png](cooke_triplet_files/cooke_triplet_12_1.png)
    


Looking at the spot diagram, we can see that rays are not focused at all. We can now optimize the parameter we created before, to try to find the best value for the image plane distance.


```python
import torch.optim as optim

tlm.optimize(optics,
             dim=2,
             optimizer = optim.Adam(optics.parameters(), lr=1e-1),
             sampling = {"base": 5, "object": 10, "wavelength": 3},
             num_iter=50,
).plot()

print("Final parameter value:", focal.item())
```

    [  1/50] L=  2.874 | grad norm= 1.0778049733901207
    [  4/50] L=  2.562 | grad norm= 1.0041784610124322
    [  7/50] L=  2.274 | grad norm= 0.9309103659202174
    [ 10/50] L=  2.009 | grad norm= 0.8583357418314483
    [ 13/50] L=  1.770 | grad norm= 0.7868110154862822
    [ 16/50] L=  1.555 | grad norm= 0.7167072738464924
    [ 19/50] L=  1.366 | grad norm= 0.6484026478234672
    [ 22/50] L=  1.200 | grad norm= 0.5822740180181092
    [ 25/50] L=  1.057 | grad norm= 0.5186882678975931
    [ 28/50] L=  0.937 | grad norm= 0.4579933129482106
    [ 31/50] L=  0.836 | grad norm= 0.4005091516637306
    [ 34/50] L=  0.754 | grad norm= 0.3465192194607496
    [ 37/50] L=  0.688 | grad norm= 0.2962623739459306
    [ 40/50] L=  0.637 | grad norm= 0.24992588487577727
    [ 43/50] L=  0.597 | grad norm= 0.20763982517661442
    [ 46/50] L=  0.568 | grad norm= 0.16947324193962748
    [ 49/50] L=  0.547 | grad norm= 0.13543241690457603
    [ 50/50] L=  0.542 | grad norm= 0.12499439075998718



    
![png](cooke_triplet_files/cooke_triplet_14_1.png)
    


    Final parameter value: 81.08018483095005



```python
f, _ = tlm.spot_diagram(optics, sampling=sampling, row="object", figsize=(12, 12))
```


    
![png](cooke_triplet_files/cooke_triplet_15_0.png)
    

