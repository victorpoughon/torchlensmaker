# Installation

## Installation with pip

Torch Lens Maker is a standard Python package distributed on PyPi:

```sh
pip install torch # add your torch install options here
pip install torchlensmaker
```

> [!IMPORTANT]
> PyTorch installation is typically platform dependent. Therefore torchlensmaker
> does not explicitely declare `torch` as a dependency, and you must [install it
> explicitly](https://pytorch.org/get-started/locally/).


Because it's currently in early development and changing fast, you might want to
try out the main branch directly:

```sh
pip install git+https://github.com/victorpoughon/torchlensmaker.git@main
```

You can test your installation by running this command:

```sh
python -c "import torch; import torchlensmaker"
```

## tlmviewer

The 3D web viewer widget [tlmviewer](https://github.com/victorpoughon/tlmviewer)
is written in TypeScript using ThreeJS, and is built as a standard npm package.

### tlmviewer in jupyter notebook

```python title="python"
tlm.show3d(optics, title="Landscape Lens")
```

By default, this code will request the tlmviewer library file from a CDN
(unpkg.com). This works out of the box but requires an internet connection. You
can also use a local version of the viewer by downloading `tlmviewer-{version}.js` from
[npmjs releases](https://www.npmjs.com/package/tlmviewer)
(or even building it yourself with npm if you prefer).

Place the `.js` file in the working directory of the jupyter server. When
running the notebook interactively, the python library will find it and use the
local version.

### tlmviewer in static html

tlmviewer can also be embedded in a static HTML page, provided with a json file of the scene:

```html title="HTML"
<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="UTF-8" />
        <title>tlmviewer example</title>

        <script type="module">
            import tlmviewer from 'https://unpkg.com/tlmviewer@latest';
            tlmviewer.loadAll();
        </script>

        <style>
            .tlmviewer {
                height: 600px;
                aspect-ratio: 1.5;
            }
        </style>
    </head>
    <body>
        <div class="tlmviewer" data-url="./testsEmbed/landscape.json"></div>
    </body>
</html>
```

