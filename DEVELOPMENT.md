# Torch Lens Maker development

## Increasing the version number

The version numbers for torchlensmaker (the python library) and tlmviewer (the 3D viewer) are **kept in sync**.

To make a new release, bump the version number in:

* tlmviewer: `package.json` (multiple occurences in the file)
* tlm: `pyproject.toml`
* tlm: `docs/package.json`

Remember to run `npm install` to update also `package-lock.json`.

## Using a local version of tlmviewer in notebooks

Build tlmviewer locally with:

```
npm run build
```

Create a symbolic link to the library in the working directory of the jupyter server:

```
ln -s ../tlmviewer/dist/tlmviewer-{version}.js
```

Running the notebook interactively will now load the local file.

Note that the version number when working on a dev version is typically higher
than the latest released version (it should be the next release version).

## Build the documentation locally

```
cd docs/
npm install

npm run docs:dev # dev server

npm run docs:build # production build
npm run docs:preview # preview the production build
```

## Exporting notebooks for the documentation

Python notebook are embeded into the documentation as `.md` files. To export a notebook to markdown, use the export script:

```
./scripts/export.py
```

## Using a local version of tlmviewer for a local documentation build

Update the version of tlmviewer in `docs/package.json` to point to your local checkout of tlmviewer:

```json
"dependencies": {
    "tlmviewer": "file:///home/user/path/to/tlmviewer/dist"
}
```

## Useful local commands

```sh
# Clear output from notebooks
uv run ./scripts/nbstripall.sh

# Start local notebook server
uv run jupyter notebook

# Execute a notebook
uv run jupyter nbconvert --execute --to notebook test_notebooks/demo_dispersion.ipynb

# Run test notebooks / examples
uv run pytest --nbmake test_notebooks/*.ipynb
uv run pytest --nbmake docs/src/examples/*.ipynb

# Run test notebooks in src
find src/ -type f -not -path '*/.*' -name "*.ipynb" | xargs uv run --no-sync pytest --nbmake

# Run unit tests
uv run pytest

# Run unit test, show output
uv run pytest -rP

# Run unit test, keep temporary file outputs
uv run pytest --basetemp tmp/

# Run unit tests with coverage report
uv run pytest --cov-report=term-missing  --cov=src/torchlensmaker/new_kinematics/ src/torchlensmaker/new_kinematics/tests/
```
