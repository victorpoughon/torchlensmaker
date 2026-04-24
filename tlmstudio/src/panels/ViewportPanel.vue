<script setup lang="ts">
import { ref, watch, onMounted, onUnmounted } from 'vue'
import { renderScene } from '../../../tlmviewer/src/render.ts'
import type { RenderHandle } from '../../../tlmviewer/src/render.ts'

defineOptions({ name: 'ViewportPanel' })

const props = defineProps<{
  params: { params: { scene: unknown | null }; api: unknown; containerApi: unknown }
}>()

const viewportEl = ref<HTMLElement>()
let handle: RenderHandle | null = null

function render(payload: unknown) {
  if (payload != null && viewportEl.value) {
    handle?.dispose()
    handle = renderScene(viewportEl.value, payload)
  }
}

onMounted(() => requestAnimationFrame(() => render(props.params.params.scene)))

watch(() => props.params.params.scene, render)

onUnmounted(() => {
  handle?.dispose()
})
</script>

<template>
  <div ref="viewportEl" class="tlmviewer" style="width: 100%; height: 100%;"></div>
</template>
