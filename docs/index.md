---
title: Torch Lens Maker
---

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

![GitHub commit activity](https://img.shields.io/github/commit-activity/w/victorpoughon/torchlensmaker)
![GitHub License](https://img.shields.io/github/license/victorpoughon/torchlensmaker)
![PRs-Welcome](https://img.shields.io/badge/PRs-Welcome-green?logo=ticktick&logoColor=green)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fvictorpoughon%2Ftorchlensmaker%2Frefs%2Fheads%2Fmain%2Fpyproject.toml&logo=Python&logoColor=yellow)
![Static Badge](https://img.shields.io/badge/PyTorch-powered-red?logo=PyTorch)
![GitHub Repo stars](https://img.shields.io/github/stars/victorpoughon/torchlensmaker?style=social)


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

tlm.show2d(optics, title="Landscape Lens")
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

### Geometric Optics

Light propagates as rays that travel in a straight line.
If they hit a surface, they can reflect, refract or stop.

### No approximations

The so-called _paraxial approximation_ ($\sin(x) = \tan(x) = x$), the _thin lens
equation_ and other approximations are very widely used in optics. In fact, I've
been frustrated with how in most material on optics, it's never really clear if
approximations are used or not, making learning difficult. Everything in _Torch
Lens Maker_ is always geometrically accurate, up to floating point precision. We
never use the paraxial approximation, the thin-lens equation, the lens's maker
equation, ABCD matrices, or any other geometric approximation.

### Sequential mode

The order in which rays of light interact with optical elements must be known.
This requirement is heavily mitigated by the fact that everything is Python
code, so complex parameterization and genericity come very naturally. It is easy
to programatically compose and combine optical elements, thanks to the dynamic
nature of the PyTorch compute graph. In practice, _Torch Lens Maker_ code
describes a tree of optical elements, and the simulation traverses that
tree in depth first search order.

### Beautiful code

The design of a software library is extremely important. The goal of this
project is not just to design lenses and mirrors, but to enable anyone to do it
with the maximum amount of correctness, verifiability and joy. This obviously
means open-source code so you can collaborate on it with git,
verify it, modify it, and most importantly read it and understand it!

I think too often code is optimized for being easy to write, when being easy to
**read** is the most important. Bringing best in class code quality and modern
software engineering to optical systems design is part of the vision for this
project.

## Features

### Optional 2D or 3D raytracing

The definition of optical elements and optical systems is generic over the
number of dimensions (2 or 3). This is possible because, to simplify a bit, data
is stored in tensors of shape `(N, D)`, where N is a batch dimension (i.e the
number of light rays being sampled) and D is 2 or 3. Every library function is
written with dimension generic code so that it is only when actually sampling
the model that the dimension must be fixed to 2 or 3. But the same model can be
sampled or optimized in either dimension.

This enables true 2D raytracing when the optical model being defined is
symmetric around the optical axis. The simulation then happens in a single
arbitrary meridional plane, where rays are 2D and don't have a Z coordinate. (X
is the principal axis in Torch Lens Maker).

This is useful because it allows optimization of a 3D system in 2D, decreasing
complexity. It also enables mixing and matching 2D and 3D for the same model
definition.

Some analysis, like spot diagrams, require a true 3D raytracing to simulate skew
rays. With this feature, it's therefore possible to define a system, optimize it
in 2D, and then generate full 3D spot diagrams without changing the model
definition.

Non axially symmetric system definition is still possible of course, but then 2D
raytracing is not available. You can still sample and simulate rays in a single
meridional plane of a 3D system, but this is not the same as 2D raytracing
(because meridional rays can be transformed out of plane in a non symmetric
system).

### Optional float32 support

By default, everything is double precision `float64`, but every library function
accepts a `dtype` argument to enable optional float32 mode.

### Flexible surface definition

implicit, rotational symmetric, freeform Surface framework based on diffoptics
for implicit surfaces any surface is possible freeform optics planned

Supported surfaces:

* Sphere
* Parabola
* Plane (Circular or Square in 3D)

> TODO

### State of the art optimization

Torch Lens Maker is based on PyTorch's exceptional autograd engine. This means
we can get exact derivatives of pretty much anything, and go crazy with gradient
descent. It's possible to optimize: 

* Parametric surface shape
* Surface diameter
* Any arbitrary 3D transform (spacing, position, rotation)
* Wavelength
* Index of refraction
* Any combination of the above jointly

> TODO link to examples in the list above

Any custom function can also be added to the loss function, so expressing design
contraints is very flexible.

Being based on PyTorch also enables access to state of the art optimization
algorithms, GPU support and everything from the machine learning community.

### Modern Python code

Designing a 3D mechanical / optical system with computer code instead of
clicking buttons in a GUI is an interesting approach because you get, for
"free":

* a fully parametric design system
* integration with other software components
* extensibility
* inspectability
* collaboration and sharing with version control

### Open-source

Torch Lens Maker is published under the [GPL-v3
license](https://github.com/victorpoughon/torchlensmaker/blob/main/LICENSE).
Community contributions [are welcome on
GitHub](https://github.com/victorpoughon/torchlensmaker).

### Export to 3D manufacturing format

This is somewhat experimental (mostly because I have zero optical manufacturing
experience), but there is support for exporting to 3D formats like STEP, STL,
etc.

This is based on the [build123d](https://build123d.readthedocs.io/en/latest/)
library, so any format supported there would be easy to add. My vision for the
project in this area is to add as many surfaces shapes from the 3D model format
directly to the library. This enables directly optimizing the final surface
parameterization to avoid the need to any conversion step. No point modeling an
ASphere if your manufacturing process only supports Bezier Splines. Then you
might as well model using Bezier Splines directly.

In the future (this is on the roadmap) I also want to add conversion between
different surface parameterization built in the library. This will enable
gradual increase in complexity of the surface shapes. So you could for example
start with a simple sphere, convert to a parabola, then to an asphere, and
refine the model at every step.

### Forward and inverse kinematics

When defining an optical system, you're actually defining two things:

* The sequence of optical elements, i.e in which order will the light rays hit
  the surfaces
* The sequence of mechanical elements, i.e what is the position and orientation
  of everything

The optical sequence is a simple linear sequence. But the mechanical sequence is
actually a tree, because you can define an element position relative to any of
the previous ones in the sequence. This is what happens with the simple `Gap()`
element, which just adds a simple translation along the optical axis **to the
previous element position**. But any arbitrary translation or rotation kinematic
tree is possible.

This effectively defines a forward kinematic chain. And in a sense, when
optimizing not just surface shapes, but also the transforms themselves (like gap
size or lens tilt), the library can be used as an inverse kinematic solver.

This is useful for example when optimizing the surface shape of a two lens
system, and still wanting the second lens to be at a fixed distance from the
outer edge of the first one. As in this example TODO link example here.

### Custom three.js based viewer

Visualization of system layout in 2D or 3D is provided by
[tlmviewer](https://github.com/victorpoughon/tlmviewer). I'm developing this
three.js based viewer (and the associated data format) jointly with the library.

It can be embeded in a webpage (like on the documentation website) or used
interactively inside a Jupyter Notebook.

