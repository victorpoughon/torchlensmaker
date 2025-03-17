# Torch Lens Maker development

## Increasing the version number

The version numbers for torchlensmaker (the python library) and tlmviewer (the 3D viewer) are **kept in sync**.

To make a new release, bump the version number in:

* tlmviewer: `package.json`
* tlm: `pyproject.toml` (multiple occurences in the file)
* tlm: `docs/package.json`

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

Update the tlmviewer library file in `docs/src/components/tlmviewer-{version}.js`.
