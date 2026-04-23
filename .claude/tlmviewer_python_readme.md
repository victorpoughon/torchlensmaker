# tlmviewer Python API

Python API for [tlmviewer](https://github.com/victorpoughon/tlmviewer), a 3D optical scene viewer for [torchlensmaker](https://victorpoughon.github.io/torchlensmaker/).

## Install

```bash
pip install tlmviewer
```

## Usage

Build a scene by creating a `Scene` and appending elements to its `data` list, then push it to a running `tlmserver` or save it to a JSON file.

```python
import tlmviewer as tlmv

scene = tlmv.Scene(mode="3D")
scene.data.append(tlmv.SceneTitle(title="My lens"))
scene.data.append(tlmv.SurfaceDisk(radius=5.0, matrix=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
scene.data.append(tlmv.Rays(points=[[0,0],[10,1],[20,0]], color="#ff0000", category="rays-valid"))

# Push to a running tlmserver
tlmv.push_scene(scene, host="127.0.0.1", port=8765, topic="main")

# Or save to a JSON file
tlmv.save_scene(scene, "scene.json")
```

## Scene elements

| Class | Protocol type |
|---|---|
| `AmbientLight` | `ambient-light` |
| `DirectionalLight` | `directional-light` |
| `SceneAxis` | `scene-axis` |
| `SceneTitle` | `scene-title` |
| `Arrows` | `arrows` |
| `Points` | `points` |
| `Rays` | `rays` |
| `Box3D` | `box3D` |
| `Cylinder` | `cylinder` |
| `SurfaceDisk` | `surface-disk` |
| `SurfaceLathe` | `surface-lathe` |
| `SurfaceSphereR` | `surface-sphere-r` |
| `SurfaceSag` | `surface-sag` |
| `SurfaceBSpline` | `surface-bspline` |

## Functions

| Function | Description |
|---|---|
| `push_scene(scene, *, host, port, topic)` | Push a scene to a running tlmserver |
| `save_scene(scene, path)` | Save a scene as a JSON file |
| `scene_to_json(scene)` | Convert a scene to a JSON string |
| `scene_to_dict(scene)` | Convert a scene to a plain dict |
