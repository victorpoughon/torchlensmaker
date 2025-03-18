---
# https://vitepress.dev/reference/default-theme-home-page
layout: doc
---

<LogoTitle/>
<Badges/>

Welcome to **Torch Lens Maker**, an open-source Python library for
differentiable geometric optics based on [PyTorch](https://pytorch.org/).
Currently a very experimental project, the goal is to be able to design complex
real-world optical systems (lenses, mirrors, etc.) using modern computer code
and state-of-the art numerical optimization.

```python
import torchlensmaker as tlm

optics = tlm.Sequential(
    tlm.ObjectAtInfinity(beam_diameter=10, angular_size=20),
    tlm.Gap(15),
    tlm.RefractiveSurface(tlm.Sphere(diameter=25, R=-45.0), material="BK7-nd"),
    tlm.Gap(3),
    tlm.RefractiveSurface(tlm.Sphere(diameter=25, R=tlm.parameter(-20)), material="air"),
    tlm.Gap(100),
    tlm.ImagePlane(50),
)

tlm.optimize(optics, tlm.optim.Adam(optics.parameters(), lr=5e-4), {"base": 10, "object": 5}, 100)

tlm.show2d(optics, title="Landscape Lens")
```

<TLMViewer src="./examples/landscape_files/landscape_0.json?url"/>

The core of the project is *differentiable geometric optics*: 3D collision
detection and the laws of optics implemented in [PyTorch](https://pytorch.org/).
PyTorch provides world-class automatic differentiation, and access to
state-of-the-art numerical optimization algorithms with GPU support.

The key idea is that there is a strong analogy to be made between layers of a
neural network, and optical elements in a so-called *sequential* optical system.
If we have a compound optical system made of a series of lenses, mirrors, etc.,
we can treat each optical element as the layer of a neural network. The
data flowing through this network are not images, sounds, or text, but rays of
light. Each layer affects light rays depending on its internal parameters
(surface shape, refractive material...) and following the very much non&#8209;linear
Snell's law. Inference, or the forward model, is the optical simulation where
given some input light, we compute the system's output light. Training, or
optimization, is finding the best shapes for lenses to focus light where we want&nbsp;it.


|                   |          **Neural Network**          |        **Optical system**         |
| ----------------: | :----------------------------------: | :-------------------------------: |
|          **Data** |         Images, Text, Audio          |            Light rays             |
|        **Layers** |         Conv2d, Linear, ReLU         |    Refraction, Reflection, Gap    |
| **Loss Function** | Prediction error to labeled examples | Focusing error in the image plane |


The magic is that we can pretty much use `torch.nn` and `nn.Module` directly,
stacking lenses and mirrors as if they were `Conv2d` and `ReLU`. Then, pass the
whole thing through a standard PyTorch `optimize()` to find the optimal values
for parametric surfaces, and designing lenses is surprisingly like training a
neural network! Once this is implemented, you get 'for free' the massive power
of modern open-source machine learning tooling: automatic differentiation,
optimization algorithms, composability, GPU training, distributed training, and
more.

On top of that, after having tried software like
[build123](https://build123d.readthedocs.io/en/latest/) and
[OpenSCAD](https://openscad.org/), I strongly believe that writing code is a very
powerful way to design mechanical 3D systems and this project is an exploration
of that, but for optical systems.

::: warning Experimental project!
This project is in its very early stages, I've got a [very long roadmap](/roadmap) planned
and I'm [looking for funding](/about#funding) to be able to keep working on it full time! If you
can, please consider donating, sponsoring or even hiring me! 😊💚

Also, the API **will** change without warning. A stable release is still very far in the future.
:::
