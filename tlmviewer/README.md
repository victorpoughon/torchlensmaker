# tlmviewer

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/victorpoughon/tlmviewer/node.js.yml)
![GitHub License](https://img.shields.io/github/license/victorpoughon/tlmviewer)
![NPM Version](https://img.shields.io/npm/v/tlmviewer)
![GitHub package.json version](https://img.shields.io/github/package-json/v/victorpoughon/tlmviewer)

Three.js based 3D viewer for [torchlensmaker](https://github.com/victorpoughon/torchlensmaker).

## Installation

```bash
npm install tlmviewer
```

## JavaScript API

```js
import tlmviewer from "tlmviewer";
```

### `tlmviewer.embed(container, json_data)`

Render a scene into a container element from a JSON string.

```js
const container = document.getElementById("viewer");
tlmviewer.embed(container, JSON.stringify(sceneData));
```

### `tlmviewer.load(container, url)`

Fetch a scene JSON from a URL and render it into a container element. Returns a `Promise<void>`.

```js
const container = document.getElementById("viewer");
await tlmviewer.load(container, "/scenes/my-scene.json");
```

### `tlmviewer.loadAll()`

Auto-initialize all elements with class `tlmviewer` that have a `data-url` attribute. Returns a `Promise<Promise<void>[]>`.

```html
<div class="tlmviewer" data-url="/scenes/my-scene.json"></div>
<script>
    tlmviewer.loadAll();
</script>
```

---

## Scene data format

A scene is a JSON object with the following top-level fields:

| Field | Type | Required | Description |
|---|---|---|---|
| `mode` | `"2D"` \| `"3D"` | yes | Rendering mode |
| `camera` | `"XY"` \| `"orthographic"` \| `"perspective"` | yes | Camera type |
| `data` | array | yes | List of scene elements |
| `controls` | object | no | Initial state of GUI controls |

**Camera types:**
- `"XY"` — 2D top-down orthographic, pan and zoom only. Use with `mode: "2D"`.
- `"orthographic"` — 3D orthographic projection with orbit controls.
- `"perspective"` — 3D perspective projection with orbit controls.

### `controls` object

The `controls` object sets the initial state of the viewer GUI. All fields are optional.

| Key | Type | Description |
|---|---|---|
| `color_rays` | string | Color option for both valid and output rays |
| `valid_rays` | string | Color option for valid rays |
| `blocked_rays` | string | Color option for blocked rays |
| `output_rays` | string | Color option for output rays |
| `opacity` | number (0–1) | Ray opacity |
| `thickness` | number (0.1–10) | Ray line width |
| `show_valid_rays` | boolean | Show/hide valid rays |
| `show_blocked_rays` | boolean | Show/hide blocked rays |
| `show_output_rays` | boolean | Show/hide output rays |
| `show_surfaces` | boolean | Show/hide optical surfaces |
| `show_optical_axis` | boolean | Show/hide the optical axis line |
| `show_other_axes` | boolean | Show/hide the coordinate axes helper |
| `show_kinematic_joints` | boolean | Show/hide kinematic joint points |
| `show_bounding_cylinders` | boolean | Show/hide bounding cylinders |
| `show_controls` | boolean | Show/hide the GUI controls panel |

Color option strings are one of: `"default"`, `"wavelength"`, `"wavelength (true color)"`, or any variable name present in the ray data.

---

## Scene elements

The `data` array contains a list of scene element objects. Each element has a `type` field that determines how it is rendered.

### `surface-sag`

A rotationally symmetric optical surface defined by a sag function. Rendered in both 2D and 3D.

```json
{
    "type": "surface-sag",
    "diameter": 10,
    "sag-function": { "sag-type": "spherical", "C": 0.1 },
    "matrix": [[1,0,0], [0,1,0], [0,0,1]]
}
```

| Field | Required | Description |
|---|---|---|
| `diameter` | yes | Aperture diameter |
| `sag-function` | yes | Sag function definition (see below) |
| `matrix` | yes | 3×3 (2D) or 4×4 (3D) homogeneous transform matrix |
| `clip_planes` | no | List of clipping planes as `[nx, ny, nz, c]` in surface local frame |

**Sag function types:**

| `sag-type` | Parameters | Description |
|---|---|---|
| `"spherical"` | `C` (curvature) | Spherical surface: `sag = C*r² / (1 + sqrt(1 - C²*r²))` |
| `"parabolic"` | `A`, `normalize` | Parabolic: `sag = A*r²` |
| `"conical"` | `C`, `K`, `normalize` | Conic section: `sag = C*r² / (1 + sqrt(1 - (1+K)*C²*r²))` |
| `"aspheric"` | `C`, `K`, `coefficients`, `normalize` | Conic + polynomial: `sag = conical + Σ coefficients[i]*r^(2i+2)` |
| `"sum"` | `terms` | Sum of other sag functions listed in `terms` array |
| `"xypolynomial"` | `coefficients`, `normalize` | XY polynomial (3D only): `sag = Σ coefficients[p][q] * y^p * z^q` |

---

### `surface-lathe`

A surface of revolution around the X axis, defined by a 2D profile curve. Rendered in both 2D and 3D.

```json
{
    "type": "surface-lathe",
    "samples": [[0.0, -5.0], [0.1, 0.0], [0.0, 5.0]],
    "matrix": [[1,0,0], [0,1,0], [0,0,1]]
}
```

| Field | Required | Description |
|---|---|---|
| `samples` | yes | Array of `[x, y]` profile points |
| `matrix` | yes | 3×3 (2D) or 4×4 (3D) homogeneous transform matrix |

---

### `surface-sphere-r`

A spherical cap surface defined by radius. Rendered in both 2D and 3D.

```json
{
    "type": "surface-sphere-r",
    "diameter": 10,
    "R": 20,
    "matrix": [[1,0,0], [0,1,0], [0,0,1]]
}
```

| Field | Required | Description |
|---|---|---|
| `diameter` | yes | Aperture diameter |
| `R` | yes | Radius of curvature (positive = center to the right, negative = left) |
| `matrix` | yes | 3×3 (2D) or 4×4 (3D) homogeneous transform matrix |

---

### `surface-plane`

A flat circular surface. Rendered in both 2D and 3D.

```json
{
    "type": "surface-plane",
    "radius": 5,
    "matrix": [[1,0,5], [0,1,0], [0,0,1]]
}
```

| Field | Required | Description |
|---|---|---|
| `radius` | yes | Radius of the plane |
| `matrix` | yes | 3×3 (2D) or 4×4 (3D) homogeneous transform matrix |
| `clip_planes` | no | List of clipping planes as `[nx, ny, nz, c]` in surface local frame |

---

### `rays`

A set of ray segments with optional per-ray scalar variables for color mapping. Rendered in both 2D and 3D.

```json
{
    "type": "rays",
    "points": [[0, 0, 10, 2], [0, 0, 10, -2]],
    "color": "#ffa724",
    "variables": { "field": [0.5, -0.5] },
    "domain": { "field": [-1, 1] },
    "categories": "rays-valid"
}
```

| Field | Required | Description |
|---|---|---|
| `points` | yes | Array of ray segments. Each is `[x1,y1, x2,y2]` in 2D or `[x1,y1,z1, x2,y2,z2]` in 3D |
| `color` | no | Default ray color as a CSS color string (default: `"#ffa724"`) |
| `variables` | no | Map of variable name → array of per-ray scalar values |
| `domain` | no | Map of variable name → `[min, max]` for color normalization |
| `categories` | no | Visibility category string: `"rays-valid"` (default), `"rays-blocked"`, or `"rays-output"` |

---

### `points`

A set of points rendered as small spheres. Rendered in both 2D and 3D.

```json
{
    "type": "points",
    "data": [[0, 0], [5, 2]],
    "color": "#ffffff"
}
```

| Field | Required | Description |
|---|---|---|
| `data` | yes | Array of `[x, y]` (2D) or `[x, y, z]` (3D) positions |
| `color` | no | Point color as a CSS color string (default: `"#ffffff"`) |
| `categories` | no | Array of visibility category strings (default: `[]`). Use `["kinematic-joint"]` for kinematic joint points |

---

### `arrows`

A set of arrow vectors. Rendered in both 2D and 3D.

```json
{
    "type": "arrows",
    "data": [[1, 0, 0, 5, 0, 0, 3]]
}
```

| Field | Required | Description |
|---|---|---|
| `data` | yes | Array of arrows. Each is `[dx,dy, px,py, length]` in 2D or `[dx,dy,dz, px,py,pz, length]` in 3D, where `[dx,dy,dz]` is the direction and `[px,py,pz]` is the origin |

---

### `box3D`

A wireframe box. 3D only.

```json
{
    "type": "box3D",
    "size": [10, 10, 10],
    "matrix": [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
}
```

| Field | Required | Description |
|---|---|---|
| `size` | yes | Box dimensions as `[width, height, depth]` |
| `matrix` | yes | 4×4 homogeneous transform matrix |

---

### `cylinder`

A wireframe bounding cylinder. Rendered in both 2D (as a rectangle) and 3D.

```json
{
    "type": "cylinder",
    "xmin": -1,
    "xmax": 1.27,
    "radius": 5,
    "matrix": [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
}
```

| Field | Required | Description |
|---|---|---|
| `xmin` | yes | Minimum X extent |
| `xmax` | yes | Maximum X extent |
| `radius` | yes | Cylinder radius |
| `matrix` | yes | 3×3 (2D) or 4×4 (3D) homogeneous transform matrix |

---

### `scene-axis`

An axis line through the origin. Rendered in both 2D and 3D. Hidden by default; visibility is controlled by the `axis-x`, `axis-y`, and `axis-z` categories.

```json
{
    "type": "scene-axis",
    "axis": "x",
    "length": 10,
    "color": "#e3e3e3"
}
```

| Field | Required | Description |
|---|---|---|
| `axis` | yes | Axis direction: `"x"`, `"y"`, or `"z"` |
| `length` | yes | Half-length of the axis line |
| `color` | no | Line color as a CSS color string (default: `"#e3e3e3"`) |

---

### `scene-title`

Sets the scene title displayed in the viewer overlay. Does not render any 3D geometry.

```json
{
    "type": "scene-title",
    "title": "My Scene"
}
```

| Field | Required | Description |
|---|---|---|
| `title` | yes | Title text to display |

---

### `ambient-light`

An ambient light source. Rendered in both 2D and 3D.

```json
{
    "type": "ambient-light",
    "color": "#ffffff",
    "intensity": 0.5
}
```

| Field | Required | Description |
|---|---|---|
| `color` | yes | Light color as a CSS color string |
| `intensity` | yes | Light intensity |

---

### `directional-light`

A directional light source. Rendered in both 2D and 3D.

```json
{
    "type": "directional-light",
    "color": "#ffffff",
    "intensity": 0.8,
    "position": [1, 2, 3]
}
```

| Field | Required | Description |
|---|---|---|
| `color` | yes | Light color as a CSS color string |
| `intensity` | yes | Light intensity |
| `position` | yes | Light position as `[x, y, z]` |

---

## Transform matrices

Surface elements use homogeneous transform matrices to set position and orientation:

- **2D elements** use a **3×3** matrix (row-major):
  ```json
  [[r00, r01, tx],
   [r10, r11, ty],
   [0,   0,   1 ]]
  ```
- **3D elements** use a **4×4** matrix (row-major):
  ```json
  [[r00, r01, r02, tx],
   [r10, r11, r12, ty],
   [r20, r21, r22, tz],
   [0,   0,   0,   1 ]]
  ```

---
