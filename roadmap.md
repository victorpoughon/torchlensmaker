# Torchlensmaker roadmap and TODOs


## Finger work - small tasks

* Update all examples to not use tlm.Module, except one for demoing custom module


## The big stuff

* Wavelength support



## New example notebooks ideas

* abbe number
* airy disk
* Cooke triplet
* what is an image
* https://en.wikipedia.org/wiki/Petzval_lens
* https://en.wikipedia.org/wiki/Rapid_Rectilinear
* dispersion demo, "pink floyd" prism
* pinhole camera model
* refractive telescope
* anchors demo
* lens inner vs outer thickness
* sharing shapes between surfaces
* using free parameters (offset along x)
* custom lens with manually making RefractiveSurface

* Optimizing a lens thickness
* Optimizing a lens shape
* Optimizing multiple things: lens shape, thickness, distance between lenses
* Resampling shapes

* Avoiding surface collisions during optimization / aka rays "negative collisions":
    * allow it or use regression
* Avoiding non colliding rays during optimization
* Avoiding "negative collisions" during optimization
* Regularization

* Image and object
* ray diagrams

* depth of field explanation

* focal length changes if you flip a plano convex lens (contrary to what textbook says)

## Various stuff

* handle refractive index in optical data
* lenses only given their inner refractive index
* refractive surface sets the next index of refraction and get the previous from input

* per parameter learning rate adapted to parameter scale
    * example: one parabola coefficient, one distance

*  more tests:
    lens maker equation
    setup a stack, call forward, check loss value aginst expected
    check loss is differentiable, and gradient is finite

* cleanup regression term / prior support

* Imaging applications
    * Image formation with a thin lens, equation 1/u - 1/v = 1/f

* Review and document sign convention

* Constraints on parameters

* Regularization API

* fix surface drawing during optimization

* tlm.AbsolutePosition : Fixed absolute positioning, ignore previous stack positioning
* make Lens class change the target to center of lens with an argument, i.e. anchors for Lens?
* improve piecewiseline: implement proper intersect_batch
* convert / resample between profile shapes
* faster example notebooks, improve convergence
* port pulaski code to new lib
* multiple configuration support (a new data dimension?) zoom lenses, etc.
* diffuse reflection
* chromatic aberation, wavelength support
* better plotting of parameters during optimization (vector shaped parameters, eg piecewiseline)
* inspiration from https://phydemo.app/ray-optics/
* cemented lenses: doublet, etc.

* in rendering: enable rendering single element if no focal points, some trick for last / first elements if no rays drawing

* custom tlm.Parameter to enforce float32 without need for trailing dot in user code

* support radial weighting in loss function. so residuals further from the optical axis can have less weight

* compute f number

* doc: glossary of terms
* doc: sign convention

## Evaluation plots

* Spot diagram: image of a point source / geometric PSF
* Spot diagram (chromatic): image of a point source emitting a mixture of wavelengths
* Ray error plot: focal loss as a function of 1 or 2 configuration

* add option to see number of rays at each stage
* add option to see aperture/spread of beam at each stage
* find intermediate images

* optical aberations and concepts: coma, astigmatism, petzval field curvature, etc...

* Support color_dim magnification plot
* Loss plot: plot loss function of a variable


## Rays not colliding with surface

2. OpticalSurface: add error_on_no_collision option, by default drop
3. OpticalSurface: add error_on_negative_collision, by default allow
4. Factor shape.collide() default impl in BaseShape
    Surface provides override to add transform


## Negative collisions, aka surfaces collisions

* Remove / make optional detection in OpticalSurface (negative ts is nominal)
* Add collisions points / rays origins / rays vectors accumulator lists to data pipe
* Add ts computation function to be used as regression terms during optimization


## Ranges for parameters

With clever internal reparametrization or regularization, can implement optimizable gaps with ranges.
e.g.:
- offset with maximum radial distance
- gap/shift with min/max

Can combine absolute positioning and relative positioning with constraints to make absolute with constraints


## New SVG rendering

* Add custom svg rendering, using maybe svg.py or equivalent, and a custom ipython _repr_javascript_ to allow pan / zoom etc.
* Possibly use a JS library like https://github.com/bumbu/svg-pan-zoom#demos

## 3D export and display improvements

* 3d export for bezier spline
* 3d export for piecewise line

* EITHER display 3D models with https://github.com/bernhard-42/three-cad-viewer
* OR use stock build123d viewer, customize it

* export full optical stack

## Images and Objects

add source_coordinate tensor to OpticalData

> Normalized parametric coordinate of the ray's source point on the shape that emitted the ray
> source_coordinates

add tlm.Object (= multiple point sources, varying height)
add tlm.ObjectAtInfinity (= multiple points sources at infinity, varying angles)

move loss compute to optical elements

add tlm.Image = on surface distance or multi point squared point line distance
add tlm.ImageAtInfinity

make ray error plot by histograming on source_coordinate

add render option for sources: draw rays a bit before their origins to see them better
> ray_draw_start = -1.5

## Thoughts about sampling

* elements definition are sampling free. they define the space.
* When evaluating for either rendering or optimization, then the space is sampled to create rays.

* add sampling distribution parameter? linspace, random-uniform, random-normal

piecewise line: need more rays to work

* add back_distance render option to light sources rendering (useful for object at infinity)

## Newton solver improvements

* add parameters with good default
* detect convergence before max iter

## Training improvements

detect when optimization leads to parameter out of domain with nice error message.
for example, setup an impossibly short focal length with a spherical lens

detect when optimization reaches a point where there are no rays exiting the system, make a nice error message

multiple loss function for image / focal point:
- ray point distance
- on shape distance

## Focal length measurement

* be able to measure focal lengths in different ways that make sense for optics (front, back, etc): choose a reference point on the lens
* position a lens relative to its center, or other points on the lens: front, back, center, nodes, etc.


## Parametric coordinates

t:
shape parametric coordinate
valid on shape.domain()
domain is anything the shape wants but must be:
0 indicates center
finite value for domain maximum

## Support for double precision floats
