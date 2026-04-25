# tlmviewer Rendering Performance Improvements

## Context

The tlmviewer Three.js rendering pipeline has several confirmed performance issues:
- The RAF animation loop renders unconditionally at 60fps even when nothing has changed
- `PointsElement` creates one `SphereGeometry` + `MeshBasicMaterial` + `Mesh` per point (N draw calls instead of 1)
- `RaysElement` destroys and recreates all geometry/material whenever the color option changes
- Because `setRaysColorOption` destroys the material, `gui.ts` must re-dispatch opacity and thickness after every color change — 3 separate scene traversals per single GUI interaction
- `TLMScene.dispatch()` traverses the entire scene graph on every event, even when only one element type is affected

---

## Step 1 — Demand-driven rendering

**Files:** `tlmviewer/src/app.ts`, `tlmviewer/src/scene.ts`, `tlmviewer/src/render.ts`

**Impact: HIGH | Complexity: LOW**

The animation loop currently calls `renderer.render()` every frame. The scene is static except when events fire or the camera moves. Switch to render-on-demand.

### Changes

**`scene.ts`** — add callback field and call it from `dispatch()`:
```typescript
public onSceneChanged: (() => void) | null = null;

public dispatch<K extends SceneEventType>(event: SceneEvent<K>): void {
    this.sceneGraph.traverse((child) => {
        if (child.userData instanceof SceneEntry) child.userData.onEvent(event);
    });
    this.onSceneChanged?.();
}
```

**`app.ts`** — track `needsRender` flag, set it from controls `change` event and expose `requestRender()`:
```typescript
private needsRender = true;

public requestRender(): void {
    this.needsRender = true;
}

public animate(): () => void {
    let animId: number;
    this.rig.controls.addEventListener('change', () => { this.needsRender = true; });
    const loop = () => {
        this.rig.controls.update();
        if (this.needsRender) {
            this.renderer.render(this.scene.scene, this.rig.camera);
            this.needsRender = false;
        }
        animId = requestAnimationFrame(loop);
    };
    animId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animId);
}
```

Also call `this.requestRender()` from `onWindowResize()` and `setCamera()`.

**`render.ts`** — wire `onSceneChanged` after scene and app are created (line 29):
```typescript
scene.onSceneChanged = () => app.requestRender();
```

**Note:** OrbitControls `change` event fires on each intermediate damping frame, so damped cameras work correctly. None of the current camera rigs enable damping, so this is a non-issue.

---

## Step 2 — RaysElement: preserve geometry and material across color changes

**File:** `tlmviewer/src/elements_basic/RaysElement.ts`

**Impact: HIGH | Complexity: MEDIUM**

The current `setRaysColorOption` calls `object.clear()` and rebuilds the entire `LineSegmentsGeometry` + `LineMaterial` on every color change. This is also why `gui.ts` must re-dispatch opacity and thickness — the old material is gone.

### Changes

Introduce a `RaysGroup` subclass to store persistent references:

```typescript
class RaysGroup extends THREE.Group {
    lines: LineSegments2 | null = null;
    material: LineMaterial | null = null;
}
```

Change `render()` to return a `RaysGroup` (still empty — populated by first color event):
```typescript
function render(_data: RaysData, _dim: number): THREE.Object3D {
    return new RaysGroup();
}
```

Rewrite `setRaysColorOption` to:
- On first call: build geometry from scratch and store references
- On subsequent calls: update only the color buffer attribute (or toggle `vertexColors` on the existing material)

```typescript
function setRaysColorOption(object: THREE.Object3D, data: RaysData, category: string, colorOption: ColorOption): void {
    if (data.category !== category) return;
    const g = object as RaysGroup;

    const { positions, colors, useVertexColors } = buildRayGeometry(data, colorOption);

    if (g.lines === null) {
        // First call: build geometry and material
        const geometry = new LineSegmentsGeometry();
        geometry.setPositions(positions);
        if (useVertexColors) geometry.setColors(colors);
        const material = new LineMaterial({
            color: useVertexColors ? undefined : data.color,
            linewidth: 1,
            vertexColors: useVertexColors,
            transparent: true,
            opacity: 1.0,
        });
        g.lines = new LineSegments2(geometry, material);
        g.material = material;
        g.add(g.lines);
    } else {
        // Subsequent call: update color buffer only
        const geometry = g.lines.geometry as LineSegmentsGeometry;
        if (useVertexColors) {
            geometry.setColors(colors);
        }
        if (g.material) {
            const wasVertex = g.material.vertexColors;
            g.material.vertexColors = useVertexColors;
            g.material.color.set(useVertexColors ? 0xffffff : data.color);
            if (wasVertex !== useVertexColors) g.material.needsUpdate = true; // vertexColors is a shader define
        }
    }
}
```

Factor the geometry/color building out of `makeRays` into two helpers:
- `buildRayPositions(data)` → `Float32Array` of positions
- `buildRayColors(data, colorOption)` → `{ colors: number[], useVertexColors: boolean }`

Rewrite opacity and thickness handlers to use the stored material reference directly (no traversal needed):
```typescript
setRaysOpacity: (_, object, event) => {
    const g = object as RaysGroup;
    if (g.material) g.material.opacity = event.value;
},
setRaysThickness: (_, object, event) => {
    const g = object as RaysGroup;
    if (g.material) g.material.linewidth = event.value;
},
```

**Important caveat:** `LineMaterial.vertexColors` is a GLSL `#define`, so toggling it requires `material.needsUpdate = true`. This triggers shader recompilation on the next frame but only when switching between mapped-color and default-color modes — not on every frame.

---

## Step 3 — GUI: remove redundant opacity/thickness re-dispatch

**File:** `tlmviewer/src/gui.ts`

**Impact: MEDIUM | Complexity: LOW** *(depends on Step 2)*

After Step 2, `setRaysColorOption` preserves the `LineMaterial` instance, so opacity and thickness survive color changes without being re-applied. Remove the two extra dispatches from each color `onChange` handler (lines 162–196):

```typescript
// BEFORE
controllerColorsValidRays.onChange((value: ColorOption) => {
    this.scene.dispatch({ type: "setValidRaysColor", value });
    this.scene.dispatch({ type: "setRaysOpacity", value: controllerColorsOpacity.getValue() });     // REMOVE
    this.scene.dispatch({ type: "setRaysThickness", value: controllerColorsThickness.getValue() }); // REMOVE
});

// AFTER
controllerColorsValidRays.onChange((value: ColorOption) => {
    this.scene.dispatch({ type: "setValidRaysColor", value });
});
```

Apply the same to `controllerColorsBlockedRays.onChange` and `controllerColorsOutputRays.onChange`.

**Must implement Steps 2 and 3 together** — Step 3 without Step 2 would cause opacity/thickness to reset to defaults on every color change.

---

## Step 4 — PointsElement: use InstancedMesh

**File:** `tlmviewer/src/elements_basic/PointsElement.ts`

**Impact: CRITICAL (GPU) | Complexity: LOW**

The current `render()` creates one `SphereGeometry` + `MeshBasicMaterial` + `Mesh` per point, yielding N draw calls. Replace with a single `InstancedMesh`:

```typescript
function render(data: PointsData, _dim: number): THREE.Object3D {
    const { vertices, color, radius } = data;
    const count = vertices.length;

    const geometry = new THREE.SphereGeometry(radius, 8, 8);
    const material = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.8 });
    const mesh = new THREE.InstancedMesh(geometry, material, count);

    const dummy = new THREE.Object3D();
    for (let i = 0; i < count; i++) {
        const point = vertices[i];
        dummy.position.set(point[0], point[1], point.length === 2 ? 2.0 : point[2]);
        dummy.updateMatrix();
        mesh.setMatrixAt(i, dummy.matrix);
    }
    mesh.instanceMatrix.needsUpdate = true;

    const group = new THREE.Group();
    group.add(mesh);
    return group;
}
```

The existing `setCategoryVisibility` handler toggles `object.visible` on the group — this still works without modification. Result: N points → 1 draw call.

---

## Step 5 — Event dispatch registry (avoid full scene traversal)

**File:** `tlmviewer/src/scene.ts`

**Impact: MEDIUM | Complexity: MEDIUM**

`TLMScene.dispatch()` currently traverses the entire `sceneGraph` on every event. Build an index from event type → `[SceneEntry]` at element-add time, so `dispatch()` only visits relevant entries:

```typescript
private eventRegistry = new Map<SceneEventType, SceneEntry[]>();

public addSceneElement(dim: number, elementData: any) {
    // ... existing parse/render ...
    const entry = new SceneEntry(object3d, data, descriptor);
    object3d.userData = entry;
    this.sceneGraph.add(object3d);

    // Register for event types this descriptor handles
    if (descriptor.events) {
        for (const eventType of Object.keys(descriptor.events) as SceneEventType[]) {
            const list = this.eventRegistry.get(eventType) ?? [];
            list.push(entry);
            this.eventRegistry.set(eventType, list);
        }
    }
}

public dispatch<K extends SceneEventType>(event: SceneEvent<K>): void {
    for (const entry of this.eventRegistry.get(event.type) ?? []) {
        entry.onEvent(event);
    }
    this.onSceneChanged?.();
}
```

**Note:** Elements are never removed within a session (full scene rebuild on new WebSocket message), so registry cleanup is not needed.

---

## Commit order

1. **Step 1** — demand rendering (standalone, safe)
2. **Steps 2 + 3** — RaysElement refactor + GUI cleanup (must go together)
3. **Step 4** — InstancedMesh for points (standalone)
4. **Step 5** — event dispatch registry (standalone)

---

## Verification

After each step:
- Load tlmstudio, send a scene with rays + points via the Python SDK
- Profile in Chrome DevTools > Performance tab: check "GPU" and "Rendering" tracks
- **Step 1:** verify the FPS counter is idle when not interacting, spikes on camera drag
- **Steps 2+3:** change color option multiple times — confirm no geometry recreation in the Memory tab
- **Step 4:** send a scene with 1000+ points — draw call count should drop from N to 1
- Confirm GUI controls (opacity, thickness, visibility) still work correctly after all refactors
