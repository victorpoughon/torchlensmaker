# Torchlensmaker roadmap and TODOs

## Pairing ideas

* Fix piecewise line
* Fix bezier spline
* per parameter learning rate adapted to parameter scale
    * example: one parabola coefficient, one distance

## New example notebooks ideas

* anchors demo
* lens inner vs outer thickness
* sharing shapes between surfaces
* using free parameters (offset along x)
* real optical systems
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

## Various stuff, higher prio

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

## Various stuff, lower prio

* tlm.AbsolutePosition : Fixed absolute positioning, ignore previous stack positioning
* make Lens class change the target to center of lens with an argument, i.e. anchors for Lens?
* improve piecewiseline: implement proper intersect_batch
* convert / resample between profile shapes
* faster example notebooks, improve convergence
* port pulaski code to new lib
* multiple configuration support (a new data dimension?) zoom lenses, etc.
* diffuse reflection
* Aperture
* chromatic aberation, wavelength support
* better plotting of parameters during optimization (vector shaped parameters, eg piecewiseline)

## Rays not colliding with surface

2. OpticalSurface: add error_on_no_collision option, by default drop
3. OpticalSurface: add error_on_negative_collision, by default allow
4. Factor shape.collide() default impl in BaseShape
    Surface provides override to add transform

## Custom execution context

custom execution context instead of nn.Sequential
an execution returns the full (data, elements) chain

* provide full output of data at each stage
* remove need for data.previous chain
* remove for forwark hooks
* add option to see number of rays at each stage
* add option to see aperture of beam at each stage

## Negative collisions, aka surfaces collisions

* Remove / make optional detection in OpticalSurface (negative ts is nominal)
* Add collisions points / rays origins / rays vectors accumulator lists to data pipe
* Add ts computation function to be used as regression terms during optimization

## Rendering

* Replace matplotlib with custom svg, using maybe svg.py or equivalent, and a custom ipython _repr_javascript_ to allow pan / zoom etc.
* Possibly use a JS library like https://github.com/bumbu/svg-pan-zoom#demos
* EITHER display 3D models with https://github.com/bernhard-42/three-cad-viewer
* OR use stock build123d viewer, customize it

## Make the principal axis X

Option 1: Surface() also flips about the Y=X axis, swapping coordinates
Option 2: Update all shapes axes

With clever internal reparametrization or regularization, can implement optimizable gaps with ranges.
e.g.:
- offset with maximum radial distance
- gap/shift with min/max

Can combine absolute positioning and relative positioning with constraints to make absolute with constraints

* Horizontal / left to right ray diagrams

## Export 3D

* 3d export for bezier spline
