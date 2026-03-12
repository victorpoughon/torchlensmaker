# Sampling

An optical systems is defined with a model that represents a physical reality
where there are infinitely many rays of light emanating from light sources.

In practice, when simulating, we need to decide how many rays we use for
computation. This is called **sampling** and the number of rays is the number of
samples.

Sampled rays can follow a particular distribution, like wavelength can be
between 400 and 600nm only, angular distribution can be restricted to &plusmn;10°,
and so on. But those distributions are continuous, and samples are a discrete realization.

* `pupil` dimension: Dimension along which we sample when all other variables are fixed.
  * If the light source is at finite distance, the pupil dimension is angular along the angular size of the emitted light beam.
  * If the light source is at infinity, the pupil dimension is linear along the diameter of the emitted light beam.

* `field` dimension: Dimension that represents the object size. The coordinate of a ray along the `field` dimension is the coordinate of its origin point on the light source.
  * If the light source is at finite distance, the field dimension is linear along the object size.
  * If the light source is at infinity, the field dimension is angular along the object apparent angular size.

* `wavel` dimension: This dimension represents wavelength.

## Sampling configuration

Sampling configuration is defined by parameters of the [light source](./light_sources) element and is specific to 2D or 3D. You can also use `set_sampling2d` and `set_sampling3d` utility functions directly:

```python
optics.set_sampling2d(wavel=10)
optics.set_sampling3d(pupil=10, field=150)
```
