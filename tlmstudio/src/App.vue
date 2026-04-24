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

onMounted(() => {
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

  ws.onerror = () => addLog('WebSocket error')
  ws.onclose = () => addLog('Disconnected from tlmserver')
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
  background: #1a1a1a;
  color: #ccc;
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
  border: 1px solid #333;
}

.viewport-panel {
  flex: 1;
  min-width: 0;
}

.log-panel {
  width: 320px;
  flex-shrink: 0;
  border-left: 1px solid #333;
}

.panel-header {
  background: #252525;
  color: #888;
  font-size: 11px;
  padding: 4px 10px;
  border-bottom: 1px solid #333;
  text-transform: uppercase;
  letter-spacing: 0.05em;
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
}

.log-entry {
  display: flex;
  gap: 8px;
  padding: 2px 0;
  font-size: 12px;
  line-height: 1.5;
  border-bottom: 1px solid #222;
}

.log-time {
  color: #555;
  flex-shrink: 0;
}

.log-text {
  color: #aaa;
  word-break: break-all;
}

.log-empty {
  color: #444;
  font-size: 12px;
  padding: 8px 0;
  font-style: italic;
}
</style>
