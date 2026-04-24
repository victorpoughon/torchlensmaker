<script setup lang="ts">
import { ref, onMounted, onUnmounted, nextTick } from 'vue'
import { renderScene } from '../../tlmviewer/src/render.ts'
import type { RenderHandle } from '../../tlmviewer/src/render.ts'
import type { Envelope } from 'tlmprotocol'

interface LogEntry {
  time: string
  text: string
}

const viewportEl = ref<HTMLElement>()
const logEntries = ref<LogEntry[]>([])
const logEl = ref<HTMLElement>()

let handle: RenderHandle | null = null
let ws: WebSocket | null = null

function addLog(text: string) {
  const time = new Date().toLocaleTimeString()
  logEntries.value.push({ time, text })
  nextTick(() => {
    if (logEl.value) logEl.value.scrollTop = logEl.value.scrollHeight
  })
}

function handleEnvelope(envelope: Envelope) {
  if (envelope.type === 'scene') {
    addLog(`Scene received (topic: ${envelope.topic})`)
    if (viewportEl.value) {
      handle?.dispose()
      handle = renderScene(viewportEl.value, envelope.payload)
    }
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

onMounted(() => {
  const workspaceUrl = (document.getElementById('app') as HTMLElement | null)
    ?.dataset.workspace

  if (workspaceUrl) {
    loadWorkspace(workspaceUrl)
  } else {
    connectWebSocket()
  }
})

onUnmounted(() => {
  ws?.close()
  handle?.dispose()
})
</script>

<template>
  <div class="studio">
    <div class="panel viewport-panel">
      <div class="panel-header">Viewport</div>
      <div ref="viewportEl" class="viewport-container tlmviewer"></div>
    </div>
    <div class="panel log-panel">
      <div class="panel-header">Log</div>
      <div ref="logEl" class="log-entries">
        <div v-for="(entry, i) in logEntries" :key="i" class="log-entry">
          <span class="log-time">{{ entry.time }}</span>
          <span class="log-text">{{ entry.text }}</span>
        </div>
        <div v-if="logEntries.length === 0" class="log-empty">Waiting for messages…</div>
      </div>
    </div>
  </div>
</template>

<style>
* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  background: #0d0d0d;
  color: #e8e8e8;
  font-family: monospace;
  height: 100vh;
  overflow: hidden;
}

.studio {
  display: flex;
  height: 100vh;
  width: 100vw;
}

.panel {
  display: flex;
  flex-direction: column;
  border: 1px solid #444;
}

.viewport-panel {
  flex: 1;
  min-width: 0;
}

.log-panel {
  width: 320px;
  flex-shrink: 0;
  border-left: 2px solid #444;
}

.panel-header {
  background: #1e1e1e;
  color: #b0b0b0;
  font-size: 11px;
  padding: 5px 10px;
  border-bottom: 1px solid #444;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  flex-shrink: 0;
}

.viewport-container {
  flex: 1;
  width: 100%;
  height: 100%;
  min-height: 0;
  position: relative;
  display: block;
}

.log-entries {
  flex: 1;
  overflow-y: auto;
  padding: 6px;
  min-height: 0;
  background: #111;
}

.log-entry {
  display: flex;
  gap: 8px;
  padding: 3px 0;
  font-size: 12px;
  line-height: 1.5;
  border-bottom: 1px solid #2a2a2a;
}

.log-time {
  color: #666;
  flex-shrink: 0;
}

.log-text {
  color: #d4d4d4;
  word-break: break-all;
}

.log-empty {
  color: #555;
  font-size: 12px;
  padding: 8px 0;
  font-style: italic;
}
</style>
