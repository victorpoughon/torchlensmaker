# Getting Started

Torch Lens Maker is code based. The viewer component, called
[tlmviewer](https://github.com/victorpoughon/tlmviewer), but it's only used for
interactive visualization, not for inputing any data.

In general, the workflow to design an optical system is:

* Write some python code that defines an optical model
* Sample the model with a discrete number of light rays along each available dimension
* Define a loss function and optimize the model parameters
* Analyse the resulting model samples

## Using Jupyter Notebooks

Jupyter notebooks are the prefered way to use Torch Lens Maker. After
[installation](/doc/installation), You should be able to run this code in a jupyter
environment, and see a biconvex spherical lens:


```python
import torchlensmaker as tlm

surface = tlm.SphereByRadius(diameter=50, R=60)
optics = tlm.lenses.symmetric_singlet(surface, tlm.OuterGap(0.2), material="BK7")
tlm.show3d(optics)
```


<TLMViewer src="./getting-started_files/getting-started_0.json?url" />


## Embedding in HTML

Jupyter is not a hard requirement. Code can also be run standalone and produce a
JSON formatted dscription of a system, that can be read by tlmviewer in static
HTML:

```python
import torchlensmaker as tlm

surface = tlm.SphereByRadius(diameter=50, R=60)
optics = tlm.lenses.symmetric_singlet(surface, tlm.OuterGap(0.2), material="BK7")
tlm.export_json(optics, "lens.json")
```

And then tlmviewer can be loaded manually in HTML to render the JSON file.

