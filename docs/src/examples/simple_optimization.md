# Simple lens optimization


```python
import torchlensmaker as tlm

surface = tlm.SphereByCurvature(diameter=15, C=1./25, trainable=True)
lens = tlm.lenses.symmetric_singlet(surface, tlm.OuterGap(1.5), material="BK7")

optics = tlm.Sequential(
    tlm.PointSourceAtInfinity(beam_diameter=18.5),
    tlm.Gap(10),
    lens,
    tlm.Gap(30),
    tlm.FocalPoint(),
)

for name, p in optics.named_parameters():
    print(name, p)

tlm.show(optics, dim=2)
tlm.show(optics, dim=3, pupil=100)
```

    2.sequence.0.surface.C Parameter containing:
    tensor(0.0400, requires_grad=True)



<TLMViewer src="./simple_optimization_files/simple_optimization_0.json?url" />



<TLMViewer src="./simple_optimization_files/simple_optimization_1.json?url" />



```python
optics.set_sampling2d(pupil=10)

tlm.optimize(
    optics,
    optimizer = tlm.optim.Adam(optics.parameters(), lr=1e-3),
    dim = 2,
    num_iter = 60
).plot()

print("Final arc radius:", 1/surface.C.item())
print("Outer thickness:", lens.outer_thickness())
print("Inner thickness:", lens.inner_thickness())

tlm.show(optics, dim=2)
tlm.show(optics, dim=3, pupil=100)
```

    [  1/60] L= 1.56896 | grad norm= 161.1410369873047
    [  4/60] L= 1.09283 | grad norm= 156.91146850585938
    [  7/60] L= 0.62931 | grad norm= 153.17071533203125
    [ 10/60] L= 0.18310 | grad norm= 133.4958038330078
    [ 13/60] L= 0.21412 | grad norm= 147.20242309570312
    [ 16/60] L= 0.36245 | grad norm= 146.2548065185547
    [ 19/60] L= 0.31415 | grad norm= 146.56021118164062
    [ 22/60] L= 0.14089 | grad norm= 147.6809844970703
    [ 25/60] L= 0.12372 | grad norm= 82.12677001953125
    [ 28/60] L= 0.19185 | grad norm= 133.56570434570312
    [ 31/60] L= 0.16637 | grad norm= 82.64508056640625
    [ 34/60] L= 0.10851 | grad norm= 8.301421165466309
    [ 37/60] L= 0.11468 | grad norm= 8.708842277526855
    [ 40/60] L= 0.12953 | grad norm= 147.75596618652344
    [ 43/60] L= 0.11523 | grad norm= 8.744029998779297
    [ 46/60] L= 0.11271 | grad norm= 8.582929611206055
    [ 49/60] L= 0.11032 | grad norm= 8.424992561340332
    [ 52/60] L= 0.10804 | grad norm= 8.268692016601562
    [ 55/60] L= 0.10787 | grad norm= 8.256511688232422
    [ 58/60] L= 0.10900 | grad norm= 8.335291862487793
    [ 60/60] L= 0.10929 | grad norm= 8.354530334472656



    
![png](simple_optimization_files/simple_optimization_2_1.png)
    


    Final arc radius: 33.372520476664235
    Outer thickness: tensor(1.5000, grad_fn=<SelectBackward0>)
    Inner thickness: tensor(3.2074, grad_fn=<SelectBackward0>)



<TLMViewer src="./simple_optimization_files/simple_optimization_2.json?url" />



<TLMViewer src="./simple_optimization_files/simple_optimization_3.json?url" />

