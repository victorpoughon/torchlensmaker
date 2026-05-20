# Visibility Control Redesign

## Decisions

| Question | Answer |
|---|---|
| Category model | Auto-discovery — categories are freeform tags on elements, discovered at scene parse time |
| Persistence across reloads | None — new scene push resets visibility to scene defaults |
| Python live commands | No — Python sets initial state via `Scene(controls=...)` only |
| Targeting | VisibilityPanel is hard-linked 1:1 to a specific ViewportPanel |
| Dispatch mechanism | In-process — panel calls tlmviewer's dispatch directly, no WebSocket round-trip |
| Panel UX | List of individual scene elements, one row per element, each with an on/off toggle |

---

## UX: VisibilityPanel

The panel is a flat list — one row per scene element. Each row shows:
- element type (e.g. `surface-sag`, `rays`, `points`)
- category string if the element has one (e.g. `rays-valid`, `my-group`)
- a single on/off toggle (checkbox or switch)

Users tick individual elements on or off. No grouping, no category-level bulk controls.
The list reflects the scene as-built; it rebuilds from scratch on each scene load.

Example panel content for a typical scene:
```
[✓] rays          rays-valid
[✓] rays          rays-output
[ ] rays          rays-blocked
[✓] surface-sag   surface
[✓] surface-sag   surface
[ ] scene-axis    axis-x
```

---

## Architecture

```
tlmstudio
├── ViewportPanel  ←──── renders tlmviewer, holds RenderHandle (reactive ref)
│       │
│       │ button in viewport opens linked panel, passes handle ref
│       ↓
└── VisibilityPanel  ─── lists elements from handle, calls handle.setElementVisibility(id)
```

No new WebSocket message type is needed. No Python API changes needed beyond what already exists.

---

## Changes by component

### 1. tlmviewer — element identity

Each scene element needs a stable ID so the panel can address it individually.
Assign a sequential integer index as elements are added in `TLMScene.addSceneElement`.
Store it on `SceneEntry`.

### 2. tlmviewer — extend `RenderHandle`

`renderScene()` currently returns:
```ts
{ getCameraState(): CameraState, dispose(): void }
```

Add:
```ts
type SceneElementInfo = {
    id: number
    type: string        // element type string, e.g. "surface-sag", "rays"
    category: string | undefined  // freeform tag if the element has one
    visible: boolean    // current visible state after defaults and controls applied
}

getElements(): SceneElementInfo[]
setElementVisibility(id: number, visible: boolean): void
```

`getElements()` iterates `sceneGraph.children`, reads `SceneEntry` from userData, and returns
one entry per element — including default scene elements (grid, axes, etc.).
`visible` is read from `child.visible` — reflects state after all
initial dispatch (defaults + `setControlsFromJson`) has run.

`setElementVisibility` sets `object.visible` directly on the matching child node.
No need for a new event type — the panel addresses elements by ID, not by category.

The existing `setCategoryVisibility` event and `show_*` in `setControlsFromJson` are kept
for Python initial-state control; they remain the mechanism for bulk default setup.

### 3. tlmviewer — remove visibility folder from GUI

Remove the "Visible" folder and all controllers from `gui.ts`.
Keep `setDefaultGUIState` visibility dispatches (they set the initial state that
`getElements()` will later read). Keep `setControlsFromJson` `show_*` handlers, but
change them to dispatch directly instead of going through removed controllers.

### 4. tlmstudio — ViewportPanel

- Change `handle` from a plain `let` to a `ref<RenderHandle | null>(null)`
- Write to it on each `render()` call
- Add a small "Visibility" button in the panel template (positioned over the viewport)
- Button calls `openVisibilityPanel()`, which uses `containerApi.addPanel()` to open a linked
  VisibilityPanel, passing the `handle` ref as a param
- If the panel is already open, focus it instead of opening a duplicate

### 5. tlmstudio — new VisibilityPanel

A new Vue SFC that:
- Receives `handle: Ref<RenderHandle | null>` as a param
- Watches `handle.value`: on change, calls `handle.value.getElements()` to rebuild the list
- Renders one row per element: human-readable type label (via lookup table) + category label + toggle
- On toggle: calls `handle.value.setElementVisibility(id, visible)`, updates local state

### 6. tlmstudio — App.vue

- Register `VisibilityPanel` in `defineOptions({ components: { ... } })`
- No other changes needed — ViewportPanel manages the open/focus logic itself via `containerApi`

---

## Implementation order

1. `tlmviewer/src/core/types.ts` — add `id` field to `SceneEntry`
2. `tlmviewer/src/scene.ts` — assign sequential IDs; add `getElements()` method
3. `tlmviewer/src/render.ts` — add `SceneElementInfo` type, `getElements()` and `setElementVisibility()` to `RenderHandle`
4. `tlmviewer/src/gui.ts` — remove visibility folder; update `setControlsFromJson` show_* to dispatch directly
5. `tlmstudio/src/panels/ViewportPanel.vue` — reactive handle ref + visibility button
6. `tlmstudio/src/panels/VisibilityPanel.vue` — new panel with lookup table for human-readable type labels
7. `tlmstudio/src/App.vue` — register VisibilityPanel
