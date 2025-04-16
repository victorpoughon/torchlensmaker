# Roadmap

This is a big list of things I'd like to do with Torch Lens Maker. Feel free to come
[discuss it on GitHub](https://github.com/victorpoughon/torchlensmaker/discussions).

## Documentation

* Write a lot more documentation!
* Integrate pdoc to generate reference documentation
* Add "download this notebook" button
* Document sign and axis convention
* Fix rendering of build123 models
* Add page discussing total internal reflection, blocked rays and sequential mode
* Add "try it live" functionality with pyodide / jupyterlite / etc.
* Document lens inner vs outer thickness
* Custom nbconvert markdown template for clearer text output cells: https://stackoverflow.com/questions/56229030
* Document sag functions (normalization, sag algebra, how to make a custom one)

## Examples

* Non rotationally symmetric example
* Petzval lens
* https://en.wikipedia.org/wiki/Rapid_Rectilinear
* Optimizing surface and position jointly
* Astigmatism example
* Regularization example
* tlm.parameter() on inner / outer thickness
* https://pbr-book.org/3ed-2018/Camera_Models/Realistic_Cameras
* Hubble space telescope repair mission
* Grazing-incidence reflection X-ray telescope
* Microscope
* Binoculars
* Periscope
* Explanation of depth of field in photography
* Custom nn.Module to group a subsequence / parameters
* Anchors demo / kinematic chain demo
* Avoiding "negative collisions" during optimization with regularization
* Thin lens equation

## Surfaces

### Sag functions
* implement more polynomials from TH thesis in both python and ts
* SagProduct: product of two sag functions (see TH thesis)
* OffsetYZ sag function transform
* Operators syntactic sugar
* ConicalSoftPlus: conical sag function without invalid domain, reparameterized with softplus

### General surfaces

* Cubic Bezier Spline
* Linear Spline
* Implicit version of SphereR
* Convert between surface types by fitting samples
* Improve `Outline` support
* use rtol / atol in `contains()` instead of bare tol
* rename `samples()` epsilon argument to clarify what it does
* implementation of parabola without Sag but direct solve of the quadratic collision equation

## Materials

* Load data from material database

## Lenses

* Cemented lenses class
* Anchors for lenses

## Optical modeling

* make focal elements "collide" rays and update them, so that `end=` works as with other elements
* research composing distance functions https://iquilezles.org/articles/distfunctions/, could be useful to model astigmatism and viewing glasses for example
* different ray category for non colliding with surface and total internal -> better blocked visualization
* tlm.parameter() on surface diameter
* tlm.parameter() on index of refraction
* F-number calculation
* Aperture detection and sampling marginal rays
* AbsolutePosition: Fixed absolute positioning
* ChromaticFilter: Filter rays based on wavelength
* OpticalSurface
    * add error_on_no_collision option, by default drop
    * add error_on_negative_collision, by default allow
* Collision detection: detect convergence before max iter with tolerance parameter
* Zoom lenses: sampling dimension for mechanical configuration
* Diffuse reflection / partial reflection (analogy to NN dropout)
* Utility function to get available dimensions of a model

## Optimization

* More tweaking options for loss function: radial weighting, wavelength weighting, etc.
* LR scheduler
* Caching of best minimum
* Optimization with 3D rays (image loss math)
* Per parameter learning rate adapted to parameter scale. Example: one parabola coefficient, one distance
* detect when optimization leads to parameter out of domain with nice error message for example, setup an impossibly short focal length with a spherical lens
* detect when optimization reaches a point where there are no rays exiting the system, make a nice error message
* better plotting of parameters during optimization (vector shaped parameters, eg piecewiseline)
* Plot of surface shape evolution during optimization
* Regularization and loss functions: Negative collision regularization (regularization on t value of collision)
* Constraints on parameters:
    * Min / Max constraint on coordinate
    * Min / Max constraint on derived coordinate: distance from principal axis (radial)
    * Combine absolute positioning and relative positioning with constraints to make absolute with constraints
    * Implement constraints with reparameterizaion or gradient projection

## Analysis

* Virtual optical element to visualize ray distribution
* Debug optical element
* Show airy disk in spot diagram
* Ray error plot: Loss as a function of 1 or 2 variables, histograming on a variable / show ray distribution over some variables
* Loss plot, plot loss function of 1 or 2 variables
* Sequence analysis:
    * Number of valid rays at each stage
    * Angular aperture/spread of beam at each stage
* Focal length measurement
    * principal points, optical center, etc.
    * be able to measure focal lengths in different ways that make sense for optics (front, back, etc: choose a reference point on the lens
    * position a lens relative to its center, or other points on the lens: front, back, center, nodes, etc.
* Find intermediate/virtual images
* Measure optical aberations and concepts: coma, astigmatism, petzval field curvature, etc...

## Manufacturing

* Implement 3D export for SphereR
* Export full stack

## TLMVIEWER

* option to view sag surface x coordinate
* option to view sag surface gradient (color by dot product with X unit vector)
* rewrite asphere glsl vertex shader to unwrapped style (no pow)
* ignore rays when computing camera auto zoom
* set camera to XY / YZ / XZ
* UI ideas from "TLM 3D surfaces rendering ideas" obsidian note
* Better rendering of Aperture
* Better rendering of ImagePlane
* Better rendering of surfaces in 3D
* Rename end= argument
* Smart non-zero default for end= argument if no focal element
* improve default sampling dict for show() function
* lens cosmetic edge
* Show/Hide cosmetic edges
* dont show legend div if title is empty
* Support multiple scenes (2D / 3D / more)
* add option for light sources: draw rays a bit before their origins to see them better
* Group elements:
    * Support arbitrary grouping of elements
    * Support tags in elements
    * Side bar with tree view of model
    * Toggle buttons in tree view of model
    * Toggle buttons for tags
* Animate rays
* Animate camera
* Slider to display a % of total number of rays
* Display stack info per layer (number of rays, aperture diameter, etc.)
* Camera selection button
* Set camera view button (XY / XY / XZ)
* Display scale bar
* Surface tooltip (material / type / parameters)
* Fullscreen button
* Support object and base coordinate color rendering for rays in 3D (use [2D colormaps](https://dominikjaeckle.com/projects/color2d/)?)
* Better handle of window resize with threejs `onWindowResize()`
* "tlmviewer loaded" should show version
* clean up/reorganize code layout in tlmviewer.py / render_sequence.py
* clean up code in scene.ts
* fix visible inner disk when rendering two half spheres
* timeline support (slider for iterations, ray variable, zoom factor)

## Testing

* test_notebooks: test sag sum in collision detection demo
* unit tests: add sagsum and other sag functions to surfaces unit tests
* unit test: test sag functions
* optical elements unit tests:
    alone, and as part of a stack
    setup a stack, call forward, check loss value aginst expected
    check loss is differentiable, and gradient is finite
* test_local_collide: add tangent test cases specific to surfaces*
* Fix collision tests with rays that have `V_X == 0`
* same sign for offset in generators for sphere and sphereR
* faster unit tests


## Advanced topics

* Enforce `mypy --strict` and ruff
* Think about `float128` support
* GPU performance benchmark
* Reduce number of project dependencies (scipy, colorcet at least)
