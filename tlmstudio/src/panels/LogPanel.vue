<script setup lang="ts">
import { ref, nextTick, watch } from 'vue'
import type { Ref } from 'vue'

export interface LogEntry {
  time: string
  text: string
}

defineOptions({ name: 'LogPanel' })

const props = defineProps<{
  params: { params: { entries: Ref<LogEntry[]> }; api: unknown; containerApi: unknown }
}>()

const logEl = ref<HTMLElement>()

watch(
  () => props.params.params.entries.value.length,
  () => nextTick(() => {
    if (logEl.value) logEl.value.scrollTop = logEl.value.scrollHeight
  }),
)
</script>

<template>
  <div ref="logEl" class="log-entries">
    <div v-for="(entry, i) in params.params.entries.value" :key="i" class="log-entry">
      <span class="log-time">{{ entry.time }}</span>
      <span class="log-text">{{ entry.text }}</span>
    </div>
    <div v-if="params.params.entries.value.length === 0" class="log-empty">Waiting for messages…</div>
  </div>
</template>

<style scoped>
.log-entries {
  height: 100%;
  overflow-y: auto;
  padding: 6px;
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
