# Torchlensmaker roadmap and TODOs

## Pairing ideas

* Fix piecewise line
* Add tests for refraction functions
* Test raytracing.py and batch raytracing functions
* Implement reflection
* Implement rotation in Surface()

## New example notebooks ideas

* Simpler version of pulaski stack: 
    * Two lenses: plano and symmetric, two different shapes but regularize for equal inner thickness or equal param

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

## Small TODOs

* Get all notebooks working
* per parameter learning rate adapted to its scale


## Rays not collisind with surface

Support different modes:
* Error: immediately error if a ray doesn't collide with surface
* extrapolate: extrapolate the surface if possible
* clamp: clamp to collision to the nearest point


## Negative collisions, aka surfaces collisions

* Remove detection in RefractiveSurface (negative ts is nominal)
* Add collisions points / rays origins / rays vectors accumulator lists to data pipe
* Add ts computation function to be used as regression terms during optimization


## Higher prio

*  more tests:
    lens maker equation
    focal loss for concave lens (i.e. negative ray position)
    setup a stack, call forward, check loss value aginst expected
    check loss is differentiable, and gradient is finite

* review / clean up optics.py
* cleanup regression term / prior support
* Imaging applications
    * Image formation with a thin lens, equation 1/u - 1/v = 1/f

* ReflectiveSurface

* Review and document sign convention

## Rendering

* Replace matplotlib with custom svg, using maybe svg.py or equivalent, and a custom ipython _repr_javascript_ to allow pan / zoom etc.
* Possibly use a JS library like https://github.com/bumbu/svg-pan-zoom#demos
* EITHER display 3D models with https://github.com/bernhard-42/three-cad-viewer
* OR use stock build123d viewer, customize it

## Lower prio

* improve piecewiseline: implement proper intersect_batch
* convert / resample between profile shapes
* faster example notebooks, improve convergence
* port pulaski code to new lib
