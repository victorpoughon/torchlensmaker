<script setup lang="ts">
import { ref, onUnmounted } from 'vue'
import { DockviewVue } from 'dockview-vue'
import type { DockviewReadyEvent } from 'dockview-core'
import 'dockview-core/dist/styles/dockview.css'
import type { Envelope } from 'tlmprotocol'
import LogPanel from './panels/LogPanel.vue'
import ViewportPanel from './panels/ViewportPanel.vue'
import type { LogEntry } from './panels/LogPanel.vue'

// Explicit registration so dockview-vue can find components by name and
// so the bundler doesn't tree-shake them away (they're not in the template).
defineOptions({ components: { ViewportPanel, LogPanel } })

const logEntries = ref<LogEntry[]>([])
const currentScene = ref<unknown | null>(null)

let ws: WebSocket | null = null

function addLog(text: string) {
  const time = new Date().toLocaleTimeString()
  logEntries.value.push({ time, text })
}

function handleEnvelope(envelope: Envelope) {
  if (envelope.type === 'scene') {
    addLog(`Scene received (topic: ${envelope.topic})`)
    currentScene.value = envelope.payload
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
  const api = event.api

  api.addPanel({
    id: 'viewport',
    component: 'ViewportPanel',
    title: 'Viewport',
    params: { scene: currentScene },
  })

  api.addPanel({
    id: 'log',
    component: 'LogPanel',
    title: 'Log',
    params: { entries: logEntries },
    position: { direction: 'right', referencePanel: 'viewport' },
    initialWidth: 320,
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
  <DockviewVue
    class="dockview-theme-dark"
    style="height: 100vh; width: 100vw;"
    @ready="onReady"
  />
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
