---
title: Torch Lens Maker
hide:
    #- navigation
    - toc
---

<style>
div.md-content {
  max-width: 900px;
  /* margin: 0 auto; */

}
</style>

<script type="module">
const module = await import("/tlmviewer.js");
const tlmviewer = module.tlmviewer;

window.addEventListener("load", (event) => {
  tlmviewer.loadAll();
});
</script>

<div id="logo-title">
<h1>Torch Lens Maker</h1>
</div>

Welcome to **Torch Lens Maker**, an open-source Python library for geometric
optics based on [PyTorch](https://pytorch.org/). Currently a very experimental
project, the ultimate goal is to be able to design complex real-world optical
systems (lenses, mirrors, etc.) using modern computer code and state-of-the art
numerical optimization.

```python
import torchlensmaker as tlm

optics = tlm.Sequential(
    tlm.ObjectAtInfinity(beam_diameter=10, angular_size=20),
    tlm.Multichromatic([450, 600, 750]),
    tlm.Gap(15),
    tlm.RefractiveSurface(tlm.Sphere(diameter=25, r=-45.759), material="BK7"),
    tlm.Gap(3.419),
    tlm.RefractiveSurface(tlm.Sphere(diameter=25, r=-24.887), material="air"),
    tlm.Gap(97.5088),
    tlm.ImagePlane(50),
)

tlm.show(optics, dim=2)
```

<div class="tlmviewer" data-url="/examples/landscape.json"></div>

The core of the project is differentiable geometric optics: 3D collision
detection and the laws of optics implemented in [PyTorch](https://pytorch.org/).
PyTorch provides world-class automatic differentiation, and access to
state-of-the-art numerical optimization algorithms with GPU support.

The key idea is that there is a strong analogy to be made between layers of a
neural network, and optical elements in a so-called "sequential" optical system.
If we have a compound optical system made of a series of lenses, mirrors, etc.,
we can pretend that each optical element is the layer of a neural network. The
data flowing through this network are not images, sounds, or text, but rays of
light. Each layer affects light rays depending on its internal parameters
(surface shape, refractive material...) and following the very much non-linear
Snell's law. Inference, or the forward model, is the optical simulation where
given some input light, we compute the system's output light. Training, or
optimization, is finding the best shapes for lenses to focus light where we want
it.

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
[build123](https://build123d.readthedocs.io/en/latest/) or
[OpenSCAD](https://openscad.org/) I strongly believe that writing code is a very
powerful way to design mechanical 3D systems and this project is an exploration
of that, but for optical systems.

In summary, I got the idea to start this project because:

* PyTorch [autograd](https://pytorch.org/docs/stable/autograd.html) is *very, very* powerful.
* Optical design software is too often ridiculously expensive and proprietary.
* Open-source optics should exist!

The ultimate goal of _Torch Lens Maker_ is to be to _Zemax OpticStudio_ what _OpenSCAD_ is to _SolidWorks_.

## Design principles

**Geometric Optics:**
Light propagates as rays that travel in a straight line.
If they hit a surface, they can reflect, refract or stop.

**No approximations:** The so-called _paraxial approximation_ ($\sin(x) =
\tan(x) = x$), the _thin lens equation_ and other approximations are very widely
used in optics. In fact, I've been frustrated with how in most material on
optics, it's never really clear if approximations are used or not, making
learning difficult. Everything in _Torch Lens Maker_ is always geometrically
accurate, up to floating point precision. We never use the paraxial
approximation, the thin-lens equation, the lens's maker equation, ABCD matrices,
or any other geometric approximation.

**Sequential mode:** The order in which rays of light interact with optical
elements must be known. This requirement is heavily mitigated by the fact that
everything is Python code, so complex parameterization and genericity come very
naturally. It is easy to programatically compose and combine optical elements, thanks to
the dynamic nature of the PyTorch compute graph. In practice, _Torch Lens Maker_
code describes a tree of optical elements, and the simulation traverses that
tree in depth first search order.

**Beautiful code:** The design of a software library is extremely important. The
goal of this project is not just to design lenses and mirrors, but to enable
anyone to do it with the maximum amount of correctness, verifiability and joy.
This obviously means open-source code so you can collaborate on it with git,
verify it, modify it, and most importantly read it and understand it!

I think too often code is optimized for being easy to write, when being easy to
**read** is the most important. Bringing best in class code quality and modern
software engineering to optical systems design is part of the vision for this
project.
