<script setup lang="ts">
import { ref, onUnmounted } from 'vue'
import { DockviewVue } from 'dockview-vue'
import type { DockviewApi, DockviewReadyEvent } from 'dockview-core'
import 'dockview-core/dist/styles/dockview.css'
import type { Envelope } from 'tlmprotocol'
import LogPanel from './panels/LogPanel.vue'
import type { LogEntry } from './panels/LogPanel.vue'
import ViewportPanel from './panels/ViewportPanel.vue'
import SceneManagerPanel from './panels/SceneManagerPanel.vue'
import type { SceneEntry } from './types.ts'

// Explicit registration so dockview-vue can find components by name and
// so the bundler doesn't tree-shake them away (they're not in the template).
defineOptions({ components: { ViewportPanel, LogPanel, SceneManagerPanel } })

const logEntries = ref<LogEntry[]>([])
const scenes = ref<SceneEntry[]>([])

let dockviewApi: DockviewApi | null = null
let ws: WebSocket | null = null

function addLog(text: string) {
  const time = new Date().toLocaleTimeString()
  logEntries.value.push({ time, text })
}

function openViewportPanel(scene: SceneEntry) {
  if (!dockviewApi) return
  const panelId = `viewport-${scene.id}`
  const existing = dockviewApi.getPanel(panelId)
  if (existing) {
    existing.focus()
    return
  }
  dockviewApi.addPanel({
    id: panelId,
    component: 'ViewportPanel',
    title: `Viewport · ${scene.topic}`,
    params: { scene: scene.payload },
    position: { direction: 'within', referencePanel: 'viewport-live' },
  })
}

function handleEnvelope(envelope: Envelope) {
  if (envelope.type === 'scene') {
    const scene: SceneEntry = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
      topic: envelope.topic,
      timestamp: new Date(),
      payload: envelope.payload,
    }
    scenes.value.push(scene)
    dockviewApi?.getPanel('viewport-live')?.update({ params: { scene: scene.payload } })
  } else if (envelope.type === 'log') {
    addLog(`[log] ${envelope.payload}`)
  }
}

function connectWebSocket() {
  const wsUrl = `ws://${location.host}/ws`
  ws = new WebSocket(wsUrl)
  ws.onopen = () => addLog('Connected to tlmserver')
  ws.onmessage = (event) => {
    let envelope: Envelope
    try {
      envelope = JSON.parse(event.data as string)
    } catch {
      addLog('Failed to parse message')
      return
    }
    handleEnvelope(envelope)
  }
  ws.onerror = () => addLog('WebSocket error')
  ws.onclose = () => addLog('Disconnected from tlmserver')
}

async function loadWorkspace(url: string) {
  addLog(`Loading workspace: ${url}`)
  try {
    const envelopes: Envelope[] = await fetch(url).then((r) => r.json())
    for (const envelope of envelopes) {
      handleEnvelope(envelope)
    }
    addLog(`Workspace loaded (${envelopes.length} messages)`)
  } catch (err) {
    addLog(`Failed to load workspace: ${err}`)
  }
}

function onReady(event: DockviewReadyEvent) {
  dockviewApi = event.api

  const W = window.innerWidth
  const H = window.innerHeight

  dockviewApi.addPanel({
    id: 'viewport-live',
    component: 'ViewportPanel',
    title: 'Viewport',
    params: { scene: null },
  })

  dockviewApi.addPanel({
    id: 'scene-manager',
    component: 'SceneManagerPanel',
    title: 'Scenes',
    params: { scenes, openViewport: openViewportPanel },
    position: { direction: 'right', referencePanel: 'viewport-live' },
    initialWidth: Math.round(W * 0.2),
  })

  dockviewApi.addPanel({
    id: 'log',
    component: 'LogPanel',
    title: 'Log',
    params: { entries: logEntries },
    position: { direction: 'below', referencePanel: 'scene-manager' },
    initialHeight: Math.round(H * 0.45),
  })

  const workspaceUrl = (document.getElementById('app') as HTMLElement | null)?.dataset.workspace
  if (workspaceUrl) {
    loadWorkspace(workspaceUrl)
  } else {
    connectWebSocket()
  }
}

onUnmounted(() => {
  ws?.close()
})
</script>

<template>
  <DockviewVue class="dockview-theme-dark" style="height: 100vh; width: 100vw" @ready="onReady" />
</template>

<style>
* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  background: #0d0d0d;
  font-family: monospace;
  height: 100vh;
  overflow: hidden;
}
</style>
