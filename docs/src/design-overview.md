---
outline: [1, 3]
---

# Design Overview

Torch Lens Maker implements <em class="hl-green">geometric optics</em>. Light is
made up of a finite number of individual rays. Rays are each modeled as an
infinite straight line in 3D Euclidean space. If they hit a surface, they can
reflect, refract or stop.

## Design principles

### No approximations

The so-called _paraxial approximation_ ($\sin(x) = \tan(x) = x$), the _thin lens
equation_ and other approximations are very widely used in optics. In fact, I've
been frustrated with how in most material on optics, it's never really clear if
approximations are used or not, making learning difficult. Everything in _Torch
Lens Maker_ is always geometrically and physically accurate, up to floating
point precision. I will never use the paraxial approximation, the thin-lens
equation, the lens's maker equation, ABCD matrices, or any other approximation.

### Sequential mode

The order in which rays of light interact with optical elements must be known.
This requirement is heavily mitigated by the fact that everything is Python
code, so complex parameterization and genericity come very naturally. It is easy
to programatically compose and combine optical elements, thanks to the dynamic
nature of the PyTorch compute graph. For an example of this, see the [Variable
Lens Sequence](/examples/variable_lens_sequence) example. In practice, _Torch
Lens Maker_ code describes a tree of optical elements, and light rays traverse
that tree in depth first search order.

### Beautiful code

The design of a software library is extremely important. The goal of this
project is not just to design lenses and mirrors, but to enable anyone to do it
with the maximum amount of correctness, verifiability and joy. This obviously
means open-source code so you can collaborate on it with git,
verify it, modify it, and most importantly read it and understand it!

I think too often code is optimized for being easy to write, when being easy to
read is the most important. Bringing best in class code quality and modern
software engineering to optical systems design is part of the vision for this
project.

## Features

### Dimension generic code

Most of the code in the library is generic over the number of dimensions (2 or 3).

This is useful because it allows to choose between 2D or 3D, depending on the
current task, without changing the model definition. With the same model, you
can move up and down the complexity ladder at will:

* 2D raytracing with a few principal and marginal rays
* Full 3D raytracing with thousands of rays

This enables true 2D raytracing when the optical model being defined is
symmetric around the optical axis. The simulation then happens in a single
arbitrary meridional plane, where rays are 2D and don't have a Z coordinate. (X
is the principal axis in Torch Lens Maker).

Some analysis, like spot diagrams, require a true 3D raytracing to simulate skew
rays. With this feature, it's therefore possible to define a system, optimize it
in 2D, and then generate full 3D spot diagrams without changing the model
definition.

> [!NOTE]
> Non axially symmetric systems can still be modeled, but then 2D
raytracing is not available. You can still sample and simulate rays in a single
meridional plane of a 3D system, but this is not the same as 2D raytracing
(because meridional rays can be transformed out of plane in a non symmetric
system).

### Flexible surface definition

The math behind surface parameterization and collision detection code is heavily inspired by the [Wang 2022 DiffOptics](https://vccimaging.org/Publications/Wang2022DiffOptics/) paper.
For now there is support for:

* Plane
* Sphere with radius parameterization
* Sphere with curvature parameterization
* Parabola
* Generic Asphere

I'm actively planning to support:

* Cubic Bezier Spline (based on resultant implicitization)
* Linear Spline (based on [torch.searchsorted](https://pytorch.org/docs/stable/generated/torch.searchsorted.html))

Fully freeform surfaces are also possible within the math framework of the library, but support for that is longer term.

### State of the art optimization

Torch Lens Maker is based on PyTorch's exceptional autograd engine. This means
we can get exact derivatives of pretty much anything, and go crazy with gradient
descent. It's possible to optimize parametric surface shapes, 3D transform
(spacing, position, rotation) and any combination of those jointly.

Custom functions can also be added to the loss, so expressing design
contraints is very flexible. One awesome idea is to be able to propagate
manufacturing constraints all the way to the initial steps of an optical design
project.

### Open-source

Torch Lens Maker is published under the [GPL-v3
license](https://github.com/victorpoughon/torchlensmaker/blob/main/LICENSE).
Community contributions [are welcome on
GitHub](https://github.com/victorpoughon/torchlensmaker).

### Export to 3D manufacturing format

This is very experimental (mostly because I have zero optical manufacturing
experience), but there is support for exporting to 3D formats like STEP, STL,
etc. This is based on the [build123d](https://build123d.readthedocs.io/en/latest/)
library, so any format supported there would be easy to add.

My vision for the project in this area is to add as many surfaces shapes from
the 3D model format directly to the library. This enables directly optimizing
the final surface parameterization to avoid the need for any conversion step
between the design tool and all the way to the electronic controller of the
actual machine that will build the part. No point modeling with a fancy 6
coefficients asphere if your manufacturing process only supports Bezier Splines.
Then you might as well model using Bezier Splines directly.

In the future (this is on the roadmap) I also want to add conversion between
different surface parameterization built in the library. This will enable
gradual increase in complexity of the surface shapes. So you could for example
start with a simple sphere, convert to a parabola, then to a Bezier Spline, and
refine the model at every step.

### Forward and inverse kinematics

When defining an optical sequence, you're actually defining two things:

* The sequence of optical elements, i.e in which order will the light rays hit
  the surfaces
* The sequence of mechanical elements, i.e what is the position and orientation
  of everything

The optical sequence is a simple linear sequence. But the mechanical sequence is
actually a tree, because you can define an element position relative to any of
the previous ones in the sequence. This is what happens with the simple `Gap()`
element, which just adds a simple translation along the optical axis. But any
arbitrary translation or rotation kinematic tree is possible.

This effectively defines what robotics calls a [forward kinematic
chain](https://en.wikipedia.org/wiki/Forward_kinematics). When optimizing not
just surface shapes, but also the transforms themselves (like gap size or lens
tilt), the library can be effectively used as an [inverse kinematic
solver](/test_notebooks/inverse_kinematics).

This is useful for example when optimizing the surface shape of a two lens
system, and still wanting the second lens to be at a fixed distance from the
outer edge of the first one.
