# Torchlensmaker roadmap and TODOs

## Higher prio

* clean up shapes code
* implement proper PiecewiseLine.intersect_batch

* review / clean up optics.py
* move everything to absolute space to simplify rendering code


* handle rays not colliding better
* cleanup regression term / prior support
* Imaging applications

## Rendering

* Replace matplotlib with custom svg, using maybe svg.py or equivalent, and a custom ipython _repr_javascript_ to allow pan / zoom etc.
* Possibly use a JS library like https://github.com/bumbu/svg-pan-zoom#demos
* EITHER display 3D models with https://github.com/bernhard-42/three-cad-viewer
* OR use stock build123d viewer, customize it

## Lower prio


* resample between profile shapes
* faster example notebooks, improve convergence
* port pulaski code to new lib¶
