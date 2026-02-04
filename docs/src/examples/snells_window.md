# Snell's Window

Have you ever noticed that when you are underwater and you look at the surface from below, you can see the entire outside world through a narrow cone? This "window" is circled by darkness (although if you are in a swimming pool it will often reflect the pool's bottom instead).

This is an optical phenomenon known as [Snell's Window](https://en.wikipedia.org/wiki/Snell%27s_window).

![Snell's Window](./snells_window.jpeg)

*(Photo from Wikipedia)*

Let's model it in Torch Lens Maker! It's actually really simple!

We start with the optical model definition. Since optics is reversible, we'll switch thing around to simplify. It will be simply:
* A point source of light, with a very wide beam angular size. This represents the observer's field of view.
* A 2 meter gap, meaning the observer is 2m deep.
* A single refractive surface to model the ocean's surface.


```python
import torchlensmaker as tlm

optics = tlm.Sequential(
    tlm.PointSource(beam_angular_size=105, material="water"),
    tlm.Gap(2),
    tlm.RefractiveSurface(tlm.CircularPlane(15), tir="reflect", material="air"),
)
```

Note how we are setting the `tir` property of the refractive surface to `"reflect"`. This parameter controls how rays behave when total internal reflection occurs. By default, `RefractiveSurface` will consider those rays "absorbed" because they are not desired in typical optical system design. But here, total internal reflection is the whole point of Snell's Window! So we enable it with this setting.

That's it! We can view the model with tlmviewer, by sampling it along its only dimension: the base dimension. Let's make it 100 rays in 2D, and 2000 in 3D.

::: tip Note
As always, models don't have a dimension. Their definition is abtract, it's only when sampling that the number of dimensions must be fixed to 2 or 3. Here, our system is rotationally symmetric, so both 2D and 3D are available.
:::


```python
tlm.show(optics, dim=2, end=2, pupil=100)
tlm.show(optics, dim=3, end=2, pupil=2000)
```


<TLMViewer src="./snells_window_files/snells_window_0.json?url" />



<TLMViewer src="./snells_window_files/snells_window_1.json?url" />

