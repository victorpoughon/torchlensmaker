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

    [  1/50] L= 2.87443 | grad norm= 1.0778049718619838
    [  4/50] L= 2.56239 | grad norm= 1.004178459519593
    [  7/50] L= 2.27377 | grad norm= 0.9309103644628934
    [ 10/50] L= 2.00944 | grad norm= 0.8583357404101004
    [ 13/50] L= 1.76991 | grad norm= 0.7868110141016678
    [ 16/50] L= 1.55535 | grad norm= 0.7167072724997153
    [ 19/50] L= 1.36554 | grad norm= 0.6484026465160047
    [ 22/50] L= 1.19987 | grad norm= 0.5822740167518357
    [ 25/50] L= 1.05737 | grad norm= 0.5186882666748047
    [ 28/50] L= 0.93670 | grad norm= 0.4579933117715781
    [ 31/50] L= 0.83624 | grad norm= 0.40050915053625563
    [ 34/50] L= 0.75414 | grad norm= 0.34651921838567634
    [ 37/50] L= 0.68838 | grad norm= 0.296262372926638
    [ 40/50] L= 0.63686 | grad norm= 0.249925883915648
    [ 43/50] L= 0.59747 | grad norm= 0.2076398242788447
    [ 46/50] L= 0.56817 | grad norm= 0.169473241107077
    [ 49/50] L= 0.54704 | grad norm= 0.13543241613960105
    [ 50/50] L= 0.54151 | grad norm= 0.12499439001795619



    
![png](cooke_triplet_files/cooke_triplet_14_1.png)
    


    Final parameter value: 81.08018483236877



```python
f, _ = tlm.spot_diagram(optics, sampling=sampling, row="object", figsize=(12, 12))
```


    
![png](cooke_triplet_files/cooke_triplet_15_0.png)
    

