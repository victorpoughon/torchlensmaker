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

### tlmviewer in jupyter notebook

```python title="python"
tlm.show3d(optics, title="Landscape Lens")
```

By default, this code will request the tlmviewer library file from a CDN
(jsdelivr). This works out of the box but requires an internet connection. You
can also use a local version of the viewer by downloading `tlmviewer-{version}.js` from
[GitHub releases](https://github.com/victorpoughon/tlmviewer/releases)
(or even building it yourself with npm if you prefer).

Place the `.js` file in the same directory as your notebook. When running the notebook interactively,
the python library will find it and use the local version.

### tlmviewer in static html

tlmviewer can also be embedded in a static HTML page, provided with a json file of the scene:

```html title="HTML"
<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="UTF-8" />
        <title>tlmviewer example</title>

        <script type="module">
            import tlmviewer from 'https://cdn.jsdelivr.net/npm/tlmviewer/+esm';
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

