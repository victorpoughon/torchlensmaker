# side quest


```python
# mode 1:  inline / chained / affecting
# surface transform = input.transforms - anchor + scale
# output transform = input.transforms - first anchor + second anchor

# mode 2: offline / free / independent
# surface transform = input.transforms + local transform - anchor + scale
# output transform = input.transforms

# how to support absolute position on chain?

# RS(X - A) + T
# surface transform(X) = CSX - A
# surface transform = anchor1 + scale + chain
# output transform = chain + anchor1 + anchor2
```

----

PointAtInfinity
    P = LinearDimension('base', domain = beam diameter / beam cross section / hit 100% of first surface, distribution)
    V = [const spherical rotation = point apparent angular position]
    > element position = start of rays

ObjectAtInfinity
    P = PointAtInfinity.P
    V = PointAtInfinity.V [.] AngularDimension('object', domain = apparent angular shape)
    > element position = start of rays

PointAtDistance
    P = [const translation = relative point position]
    V = AngularDimension('base', domain = beam angular size / beam angular cross section / hit 100% of first surface)
    > add trick to make rays start at element position (move rays forward by some t)

ObjectAtDistance
    P = PointAtDistance.P [.] LinearDimension('object', domain = object shape)
    V = PointAtDistance.V
    > add trick to make rays start at element position (move rays forward by some t)


Additional dimensions:
    Var = Dimension('name', domain=...)


Wavelength
    const = no wavelength info
    var = wavelength in mn

Material:
    const = constant index of refraction
    var = variable index of refraction: temperature...

Refraction model inputs:
    wavelength
    material
    temperature
    any other var

---
Simple version: trivial spherical / circular domains + linspace distribution

PointAtInfinity
    P = LinearDimension('base', domain = circular beam diameter)
    V = [unit X vector]
    > element position = start of rays

ObjectAtInfinity
    P = PointAtInfinity.P
    V = AngularDimension('object', domain = object apparent angular size // all objects are circles)
    > element position = start of rays

PointAtDistance
    P = [0]
    V = AngularDimension('base', domain = beam angular size)
    > add trick to make rays start at element position (move rays forward by some t)

ObjectAtDistance
    P = LinearDimension('object', domain = object physical diameter // all objects are circles)
    V = PointAtDistance.V
    > add trick to make rays start at element position (move rays forward by some t)

---

LinearDimension(origin, domain, number of samples)
AngularDimension(origin, domain, number of samples)