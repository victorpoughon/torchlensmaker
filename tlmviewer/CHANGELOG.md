# tlmviewer changelog

## v0.0.14

### Breaking changes

- "Bounding cylinder" is now a standalone element type `"cylinder"`.

Instead of being defined as a field on surface elements. The old format embedded
cylinder data in surface elements as 

 ```
{ type: "surface-*", bcyl: [xmin, xmax, radius], matrix: ..., ... }
```

The new format is a separate element:

```
{ type: "cylinder", xmin: number, xmax: number, radius: number, matrix: number[][] }
```

- Scene title is now a `"scene-title"` element instead of a top-level `title` field.

Old format:

```json
{ "title": "My Scene", "mode": "2D", "data": [...] }
```

New format:

```json
{ "mode": "2D", "data": [..., { "type": "scene-title", "title": "My Scene" }] }
```

- Rays visibility is now controlled via `categories` instead of the `"hide"` color option.

Rays elements must include a `"categories"` field to be togglable in the GUI. The `"hide"` color option has been removed.

Old format:

```json
{ "type": "rays", "points": [...], "layers": [0] }
```

With controls: `{ "blocked_rays": "hide" }`

New format:

```json
{ "type": "rays", "points": [...], "layers": [0], "categories": ["rays-valid"] }
```

With controls: `{ "show_blocked_rays": false }`

Valid category values for rays: `"rays-valid"`, `"rays-blocked"`, `"rays-output"`.

- The `layers` field has been removed from `points` and `rays` elements.

Points and rays visibility is now controlled via `categories` and the `setCategoryVisibility` event. The `layers` field is no longer used. Kinematic joint points must include `"categories": ["kinematic-joint"]` in their data to respond to the GUI visibility toggle.

Old format:

```json
{ "type": "points", "data": [...], "layers": [4] }
```

New format:

```json
{ "type": "points", "data": [...], "categories": ["kinematic-joint"] }
```

- Camera "XY" is now called "2D".

- Camera `"axial"` is renamed to `"axial-xy"`. All 9 axis/up combinations are now available:
  `"axial-xx"`, `"axial-xy"`, `"axial-xz"`, `"axial-yx"`, `"axial-yy"`, `"axial-yz"`,
  `"axial-zx"`, `"axial-zy"`, `"axial-zz"`. The first letter is the orbit axis, the second
  is the screen-up direction.

- The `controls` keys `show_optical_axis` and `show_other_axes` are replaced by `show_axis_x`, `show_axis_y`, and `show_axis_z`.

Old format:

```json
{ "controls": { "show_optical_axis": true, "show_other_axes": false } }
```

New format:

```json
{ "controls": { "show_axis_x": true, "show_axis_y": false, "show_axis_z": false } }
```

- Element type `"surface-plane"` is renamed to `"surface-disk"`.

```json
{ "type": "surface-disk", "radius": 5, "matrix": [...] }
```