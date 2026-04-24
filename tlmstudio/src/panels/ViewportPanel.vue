<script setup lang="ts">
import { ref, watch, onUnmounted } from 'vue'
import type { Ref } from 'vue'
import { renderScene } from '../../../tlmviewer/src/render.ts'
import type { RenderHandle } from '../../../tlmviewer/src/render.ts'

defineOptions({ name: 'ViewportPanel' })

const props = defineProps<{
  params: { params: { scene: Ref<unknown | null> }; api: unknown; containerApi: unknown }
}>()

const viewportEl = ref<HTMLElement>()
let handle: RenderHandle | null = null

watch(
  () => props.params.params.scene.value,
  (payload) => {
    if (payload !== null && viewportEl.value) {
      handle?.dispose()
      handle = renderScene(viewportEl.value, payload)
    }
  },
)

onUnmounted(() => {
  handle?.dispose()
})
</script>

<template>
  <div ref="viewportEl" class="tlmviewer" style="width: 100%; height: 100%;"></div>
</template>
