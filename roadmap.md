# Torch Lens Maker - Project Roadmap


## v0.1

### TLM

* Surfaces:
    * Asphere
    * Linear Spline

* Analysis:
    * Spot diagram
    * Ray distribution visu

* Cemented lenses: doublet, etc.

* Examples:
    * Double Gauss
    * Landscape singlet (narrative)
    * Non rotationally symmetric example
    * Petzval
    * Cooke triplet
    * Ernostar 1928
    * Optimizing surface and position jointly

* Test notebooks:
    * Regularization
    * Snell's Window
    * tlm.parameter() on inner / outer thickness
    * tlm.parameter() on surface diameter

* Rendering:
    * Better rendering of Aperture
    * Better rendering of ImagePlane
    * Rename end= argument
    * Smart non-zero default for end= argument if no focal element
    * smart default sampling dict for show() function

### TLMVIEWER


* lens cosmetic edge

* Set viewer width/height from json
* Embed in static html
* Load data from external source
* Version number in build artifact
* Available on cdnjs and npm
* Show/Hide cosmetic edges
* Set default gui values from json (+python arguments to show)
* Default to object dim if it exists


### DOC

* Review and document sign convention
* List page: lenses ("simple_lenses" / cemented lenses / making your own lens)
* Embed tlmviewer
* Automatic build of examples
* Automatic build of test_notebooks (a bit hidden from main doc)


### PROJECT

* copyright / license headers
* pip install torchlensmaker
* nice roadmap

## Post v0.1

* GPU port and performance benchmarks
* f-number
* Convert between surfaces by fitting them
* tlm.parameter() on material index of refraction
* More unit tests and coverage tool
* mypy --strict

* Examples:
    * Custom nn.Module to group a subsequence / parameters
    * Custom nn.Module for advanced OpticalData manipulation
    * How rainbows work
    * https://en.wikipedia.org/wiki/Petzval_lens
    * https://en.wikipedia.org/wiki/Rapid_Rectilinear
    * https://en.wikipedia.org/wiki/Cooke_triplet
    * https://pbr-book.org/3ed-2018/Camera_Models/Realistic_Cameras
    * mars rover / hubble repair mission / space cameras
    * x ray lens with weird concentric stuff
    * Pink Floyd prism
    * Refractive telescope
    * Microscope
    * Binoculars
    * Periscope
    * Scanning mirror
    * Depth of Field in photography

* Better float32 coverage

* Improve Outline support:
    * More outlines
    * Clearer 2D/3D behavior


* Surfaces:
    * Cubic Bezier Spline

* Optimization:
    * More tweaking options for loss function: radial weighting, wavelength weighting, etc.
    * LR scheduler
    * Caching of best minimum
    * Per parameter learning rate adapted to parameter scale. Example: one parabola coefficient, one distance
    * detect when optimization leads to parameter out of domain with nice error message.
for example, setup an impossibly short focal length with a spherical lens
    * detect when optimization reaches a point where there are no rays exiting the system, make a nice error message
    * better plotting of parameters during optimization (vector shaped parameters, eg piecewiseline)
    * Plot of surface shape evolution during optimization

* Regularization and loss functions:
    * Negative collision regularization (regularization on t value of collision)

* Constraints on parameters:
    * Min / Max constraint on coordinate
    * Min / Max constraint on derived coordinate: distance from principal axis (radial)
    * Combine absolute positioning and relative positioning with constraints to make absolute with constraints
    * Implement constraints with reparameterizaion or gradient projection

* Color rays based on contribution to loss function

* Optical elements:
    * HideRays: utility to hide rays until a certain distance
    * Debug: debug info about optical data
    * AbsolutePosition: Fixed absolute positioning
    * ChromaticFilter: Filter rays based on wavelength

* OpticalSurface
    * add error_on_no_collision option, by default drop
    * add error_on_negative_collision, by default allow

* Newton's method solver:
    * detect convergence before max iter with tolerance parameter
    * better default chosen with a representative test

* Multiple configuration support (a new ray dimension) zoom lenses, etc.

* Rendering:
    * add option for light sources: draw rays a bit before their origins to see them better

### TLMVIEWER

* Animate rays

* Animate camera

* Group elements:
    * Support arbitrary grouping of elements
    * Support tags in elements
    * Side bar with tree view of model
    * Toggle buttons in tree view of model
    * Toggle buttons for tags

* Camera selection button

* Multiple scene in one file (2D / 3D toggle button or scene dropdown menu)

* Display stack info per layer (number of rays, aperture diameter, etc.)

* Slider to display a % of total number of rays


post-v1:
* nicer surfaces rendering: lights? etc.
* multiple scenes in one file
* set camera view to: XY / XZ / YZ

Layers:
- Optical Surfaces (refraction / reflection)
- Valid rays
- blocked rays
- output rays
- optical axis
- other axes
- skybox
- grid
- kinematic chain
- light source (surface marker)
- aperture (surface marker)

Surfaces tooltip:
- material
- type of surface
- parameter values


## Various stuff / brainstorming

* abbe number
* airy disk
* anchors demo
* lens inner vs outer thickness
* sharing shapes between surfaces
* using free parameters (offset along x)
* Avoiding surface collisions during optimization / aka rays "negative collisions":
    * allow it or use regression
* Avoiding non colliding rays (blocked) during optimization
* Avoiding "negative collisions" during optimization
* focal length changes if you flip a plano convex lens (contrary to what textbook says)
* optical elements unit tests:
    lens maker equation
    setup a stack, call forward, check loss value aginst expected
    check loss is differentiable, and gradient is finite
* Imaging applications
    * Image formation with a thin lens, equation 1/u - 1/v = 1/f

* make Lens class change the target to center of lens with an argument, i.e. anchors for Lens?

* faster example notebooks, improve convergence

* diffuse reflection

* inspiration from https://phydemo.app/ray-optics/

## Evaluation plots and analysis tools

* Spot diagram:
    * Chromatic support: image of a point source emitting a mixture of wavelengths

* Ray error plot:
    * Loss as a function of 1 or 2 variables
    * histograming on a variable / show ray distribution over some variables

* Loss plot:
    * plot loss function of 1 or 2 variables

* Sequence analysis:
    * Number of valid rays at each stage
    * Angular aperture/spread of beam at each stage

* Focal length measurement
    * principal points, optical center, etc.
    * be able to measure focal lengths in different ways that make sense for optics (front, back, etc: choose a reference point on the lens
    * position a lens relative to its center, or other points on the lens: front, back, center, nodes, etc.

* Find intermediate images

* Measure optical aberations and concepts: coma, astigmatism, petzval field curvature, etc...

## 3D export and display improvements

* 3d export for bezier spline
* 3d export for piecewise line
* export full optical stack
