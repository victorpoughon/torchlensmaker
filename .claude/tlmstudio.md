# tlmstudio design document

tlmstudio is a powerful studio-style viewer for optical systems made with torchlensmaker.

## Goals

- Receive scenes, logs, and plots from Python code via a relay server (tlmserver)
- Load data locally as a static HTML page (no server required)
- Provide a rich multi-panel IDE-like interface for inspecting optical systems

## Non-goals (MVP)

- Interactive parameter editing (future: once tlmtrace is available)
- Model support (future: Python sends lightweight model, tlmtrace raytraces in browser)
- Plot viewer panels (future)
- Panel layout persistence (future: localStorage via Dockview serialization)
- Topic-based panel routing (future: topics are in the message schema but not used for routing yet)

## Relationship to existing packages

tlmviewer remains a standalone library with its public API (`embed`, `load`, `loadAll`, `connect`) unchanged. tlmstudio is a consumer of tlmviewer — it embeds tlmviewer viewports inside Dockview panels. Other use cases (embedding a scene directly in a static HTML page via `tlmviewer.embed()`) remain fully supported and independent of tlmstudio.

```
tlmviewer (Python package)  →  tlmserver  →  [WebSocket]  →  tlmstudio
                                                           (Vue app, uses tlmviewer + tlmtrace)
```

## Tech stack

- Vue 3 + TypeScript
- Dockview (https://dockview.dev/) for the multi-panel layout
- tlmviewer library for 3D rendering inside viewport panels

## Workspaces

tlmstudio is a new workspace `tlmstudio/` in this monorepo. It is a private package (not published to npm). Future workspaces:
- `tlmtrace/` — TypeScript raytracer (future)

## Scene accumulation

tlmstudio treats all received data as an append-only log. Receiving multiple scene messages in sequence adds each to the scene list — nothing is silently replaced. A Python script that sends 10 scene variants produces 10 entries in the scene list. The user clears the list explicitly via a clear button. This matches the typical workflow of iterating on a design in Python and inspecting all the variants.

## Architectural principle: everything is a panel

All UI in tlmstudio is a Dockview panel. There is no global sidebar, toolbar, or chrome outside of Dockview. This keeps the code highly compartmentalized — each panel is an independent Vue component with a clear interface, and Dockview handles all layout, docking, resizing, and tab management.

## MVP panel set

Four panels for MVP:

### 1. Scene Manager
- Always-open panel listing all received scenes in order (topic, timestamp, element count, name/title)
- Clicking a scene opens a new 3D Viewport panel for that scene
- In live mode, each newly received scene also auto-opens a viewport panel
- Clear button to reset the list
- Each panel is self-contained: no shared selection state required

### 2. 3D Viewport (one per scene, opened on demand)
- Opened by clicking a scene in the Scene Manager (or auto-opened on receive in live mode)
- Embeds a tlmviewer instance for that specific scene
- Self-contained: the scene is baked in at panel creation time
- Multiple viewport panels can be open simultaneously (as Dockview tabs or side-by-side)

### 3. Scene Inspector (one per scene, opened on demand)
- Shows the element list of a specific scene with types and key properties
- Opened via a secondary action: right-click on a scene in the Scene Manager, or a small button alongside each scene entry
- Self-contained, read-only for MVP

### 4. Log Console
- Always-open panel
- Displays log messages received via tlmserver (`type: "log"`)
- Messages accumulate in chronological order
- Simple text display with timestamps
- Clear button to reset

## Data flow (live mode)

1. tlmstudio connects to tlmserver via WebSocket
2. On receiving a `type: "scene"` envelope: append to the scene list and auto-open a new viewport panel
3. On receiving a `type: "log"` envelope: append to the log console
4. Topics are preserved in the envelope but not used for panel routing yet

## Data flow (static mode)

tlmstudio can be deployed as a static HTML page (no server required). The workspace file is a JSON array of message envelopes — the same format as the live WebSocket stream. tlmstudio processes them through the same pipeline; static mode is just replaying a recorded message list.

1. Page loads, reads a `data-workspace` attribute on the tlmstudio root element to find the workspace file URL (consistent with tlmviewer's existing `data-url` convention), and fetches it
2. Envelopes are processed in order through the same pipeline as live mode
3. Panel layout is auto-determined from the message types present
4. No WebSocket connection

This makes tlmstudio embeddable as a static page (docs site, shareable link) with zero special infrastructure. A workspace file can also be produced by recording a live session.

File picker / drag-and-drop over a workspace file can be supported as an additional interaction.

### Layout auto-determination

tlmstudio infers a sensible panel layout from the message types received (from the workspace file or the first live messages):
- A `scene` message → open a 3D viewport panel
- A `log` message → open a log console panel
- An `image` message → open a plot panel (future)

Dockview has good serialization support, so explicit layout save/restore can be added later without architectural changes.

## tlmserver integration

Once tlmstudio is built, `tlmserver`'s `GET /` route should serve the tlmstudio app instead of the current minimal tlmviewer HTML page. This makes the "run tlmserver, open browser" experience land directly in the full studio UI. tlmserver will serve the tlmstudio build artifacts alongside the existing `/push` and `/ws` endpoints.

## Future: model support + tlmtrace

Once tlmtrace (TypeScript raytracer) exists:
- Python sends a lightweight model envelope (`type: "model"`) instead of a scene
- tlmstudio receives the model, calls tlmtrace to raytrace it in the browser, renders the resulting scene
- Much more lightweight over the wire; enables interactive features

Once tlmtrace is in place, interactive parameter editing becomes possible:
- The scene inspector evolves into a model inspector with editable parameters
- Changing a parameter triggers a re-trace via tlmtrace and updates the viewport in real time

## Future: topic-based panel routing

Topics are already in the message envelope schema. Future behavior TBD — options include:
- One panel per topic (auto-created)
- User manually assigns topics to panels
- A topic selector per panel

## Future: plot viewer

Receive `type: "image"` envelopes and display static images (e.g. matplotlib PNGs) in a dedicated plot panel.
