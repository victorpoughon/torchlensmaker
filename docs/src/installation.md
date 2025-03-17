# Installation

## With pip

Torch Lens Maker is a standard Python package distributed on PyPi:

```sh
pip install torchlensmaker
```

Because it's currently in early development and changing fast, you might want to
try out the main branch directly:

```sh
pip install git+https://github.com/victorpoughon/torchlensmaker.git@main
```

You can test your installation by running this command:

```sh
python -c "import torchlensmaker"
```

## tlmviewer

The 3D web viewer widget [tlmviewer](https://github.com/victorpoughon/tlmviewer)
is written in TypeScript using ThreeJS, and is built as a standard npm package.
It can be used:

* as an interactive widget from a Jupyter Notebook:

```python title="python"
tlm.show3d(optics, title="Landscape Lens")
```

* embeded statically in an HTML page:

```html title="HTML"
<div class="tlmviewer" data-url="/examples/landscape.json"></div>
```

