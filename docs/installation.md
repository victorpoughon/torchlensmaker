# Installation

TODO distribute with pip

## tlmviewer

The 3D web viewer [tlmviewer](https://github.com/victorpoughon/tlmviewer) is written in TypeScript and built as a standard npm package. It can be used:

* as an interactive widget from a Jupyter Notebook:

```python title="python"
tlm.show2d(optics, title="Landscape Lens")
```

* embeded statically in an HTML page:

```html title="HTML"
<div class="tlmviewer" data-url="/examples/landscape.json"></div>
```

TODO how to use it with tlm
