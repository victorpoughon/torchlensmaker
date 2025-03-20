# Sampling

An optical systems is defined with a model that represents a physical reality
where there are infinitely many rays of light emanating from light sources.

In practice, when simulating, we need to decide how many rays we use for
computation. This is called **sampling** and the number of rays is the number of
samples.

Sampled rays can follow a particular distribution, like wavelength can be
between 400 and 600nm only, angular distribution can be restricted to &plusmn;10°,
and so on. But those distributions are continuous, and samples are a discrete realization.

* `base` dimension: This dimension is always present unless we simulate a single ray. It's the dimension along which we sample when all other variables are fixed.

* `object` dimension: This dimension is present if the light source represents an object with a size. The coordinate of a ray along the `object` dimension is the coordinate of its origin point on the light source.

* `wavelength` dimension: This dimension is present if rays contain wavelength information. 

## The sampling dictionary

Sampling configuration is defined with a dictionary. By default, an integer value refers to dense uniform sampling, and a list of numbers refers to exact sampling at those values:


```python
# This sampling configuration will produce a total of 150 rays
sampling = {"base": 10, "object": 5, "wavelength": [600, 650, 700]}
```

For more advanced sampling configuration, use a sampling function directly:


```python
import torchlensmaker as tlm

# This sampling configuration will produce a total of 150 rays
sampling = {"base": 10, "object": tlm.random_normal(5, std=2.0), "wavelength": 3}
```

## `dense`

Uniformly spaced samples over the domain. In 2D, it is essentially `torch.linspace`. In 3D, it is [uniform disk sampling](https://victorpoughon.fr/non-random-uniform-disk-sampling/).

## `random_uniform`

Uniform random distribution

## `random_normal`

Normal distribution

## `exact`

Exact sampling at the provided list of values
