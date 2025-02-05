
## Optional 2D or 3D raytracing

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

## Optional float32 support

By default, everything is double precision `float64`, but every library function
accepts a `dtype` argument to enable optional float32 mode.

## Flexible surface definition

implicit, rotational symmetric, freeform Surface framework based on diffoptics
for implicit surfaces any surface is possible freeform optics planned

Supported surfaces:

* Sphere
* Parabola
* Plane (Circular or Square in 3D)

> TODO

## State of the art optimization

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

## Modern Python code

Designing a 3D mechanical / optical system with computer code instead of
clicking buttons in a GUI is an interesting approach because you get, for
"free":

* a fully parametric design system
* integration with other software components
* extensibility
* inspectability
* collaboration and sharing with version control

## Open-source

Torch Lens Maker is published under the [GPL-v3
license](https://github.com/victorpoughon/torchlensmaker/blob/main/LICENSE).
Community contributions [are welcome on
GitHub](https://github.com/victorpoughon/torchlensmaker).

## Export to 3D manufacturing format

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

## Forward and inverse kinematics

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

## Custom three.js based viewer

Visualization of system layout in 2D or 3D is provided by
[tlmviewer](https://github.com/victorpoughon/tlmviewer). I'm developing this
three.js based viewer (and the associated data format) jointly with the library.

It can be embeded in a webpage (like on the documentation website) or used
interactively inside a Jupyter Notebook.

