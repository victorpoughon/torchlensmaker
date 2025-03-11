---
# https://vitepress.dev/reference/default-theme-home-page
layout: doc
---

<LogoTitle/>
<Badges/>

Welcome to **Torch Lens Maker**, an open-source Python library for geometric
optics based on [PyTorch](https://pytorch.org/). Currently a very experimental
project, the ultimate goal is to be able to design complex real-world optical
systems (lenses, mirrors, etc.) using modern computer code and state-of-the art
numerical optimization.

```python
import torchlensmaker as tlm

optics = tlm.Sequential(
    tlm.ObjectAtInfinity(beam_diameter=10, angular_size=20),
    tlm.Wavelength(400, 800),
    tlm.Gap(15),
    tlm.RefractiveSurface(tlm.Sphere(diameter=25, r=-45.759), material="BK7"),
    tlm.Gap(3.419),
    tlm.RefractiveSurface(tlm.Sphere(diameter=25, r=-24.887), material="air"),
    tlm.Gap(97.5088),
    tlm.ImagePlane(50),
)

tlm.show2d(optics, title="Landscape Lens")
```

<TLMViewer src="./examples/landscape.json"/>

The core of the project is *differentiable geometric optics*: 3D collision
detection and the laws of optics implemented in [PyTorch](https://pytorch.org/).
PyTorch provides world-class automatic differentiation, and access to
state-of-the-art numerical optimization algorithms with GPU support.

The key idea is that there is a strong analogy to be made between layers of a
neural network, and optical elements in a so-called *sequential optical system*.
If we have a compound optical system made of a series of lenses, mirrors, etc.,
we can pretend that each optical element is the layer of a neural network. The
data flowing through this network are not images, sounds, or text, but rays of
light! Each layer affects light rays depending on its internal parameters
(surface shape, refractive material...) and following the very much non-linear
Snell's law. Inference, or the forward model, is the optical simulation where
given some input light, we compute the system's output light. Training, or
optimization, is finding the best shapes for lenses to focus light where we want&nbsp;it.

<div class="center-table" markdown>

|                   |          **Neural Network**          |        **Optical system**         |
| ----------------: | :----------------------------------: | :-------------------------------: |
|          **Data** |         Images, Text, Audio          |            Light rays             |
|        **Layers** |         Conv2d, Linear, ReLU         |    Refraction, Reflection, Gap    |
| **Loss Function** | Prediction error to labeled examples | Focusing error in the image plane |

</div>

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

In summary, I got the idea to start this project because:

* [torch.autograd](https://pytorch.org/docs/stable/autograd.html) is *very, very* powerful.
* Optical design software is too often ridiculously expensive and proprietary.
* Open-source optics should exist!

The ultimate goal of _Torch Lens Maker_ is to be to _Zemax OpticStudio_ what _OpenSCAD_ is to _SolidWorks_.

This project is in its very early stages, I've got a [very long roadmap](/about/#roadmap) planned
and I'm [looking for funding](/about/#funding) to be able to keep working on it full time! If you
can, please consider donating, sponsoring or even hiring me! 😊💚
