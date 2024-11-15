# Torchlensmaker roadmap and TODOs

## Higher prio

*  more tests:
    lens maker equation
    focal loss for concave lens (i.e. negative ray position)
    setup a stack, call forward, check loss value
    check loss is differentiable, and gradient is finite
    render a stack visualy (no opmitization)


* review / clean up optics.py
* move everything to absolute space to simplify rendering code

* handle rays not colliding better
* handle surfaces colliding each other better
* cleanup regression term / prior support
* Imaging applications
    * Image formation with a thin lens, equation

* VariableGap
* ReflectiveSurface

## Rendering

* Replace matplotlib with custom svg, using maybe svg.py or equivalent, and a custom ipython _repr_javascript_ to allow pan / zoom etc.
* Possibly use a JS library like https://github.com/bumbu/svg-pan-zoom#demos
* EITHER display 3D models with https://github.com/bernhard-42/three-cad-viewer
* OR use stock build123d viewer, customize it

## Lower prio

* improve piecewiseline: implement proper intersect_batch
* convert / resample between profile shapes
* faster example notebooks, improve convergence
* port pulaski code to new lib¶
