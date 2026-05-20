<script setup lang="ts">
import { ref, watch, onMounted, onUnmounted } from 'vue'
import { renderScene } from '../../../tlmviewer/src/render.ts'
import type { RenderHandle } from '../../../tlmviewer/src/render.ts'
import type { DockviewApi } from 'dockview-core'

defineOptions({ name: 'ViewportPanel' })

const props = defineProps<{
    params: { params: { scene: unknown | null }; api: any; containerApi: unknown }
}>()

const viewportEl = ref<HTMLElement>()
const handle = ref<RenderHandle | null>(null)

function render(payload: unknown) {
    if (payload != null && viewportEl.value) {
        handle.value?.dispose()
        handle.value = renderScene(viewportEl.value, payload)
    }
}

function openVisibilityPanel() {
    const containerApi = props.params.containerApi as DockviewApi
    const panelId = props.params.api?.id as string
    const visibilityPanelId = `visibility-${panelId}`

    const existing = containerApi.getPanel(visibilityPanelId)
    if (existing) {
        existing.focus()
        return
    }

    containerApi.addPanel({
        id: visibilityPanelId,
        component: 'VisibilityPanel',
        title: 'Visibility',
        params: { handle },
        position: { direction: 'within', referencePanel: 'log' },
    })
}

onMounted(() => requestAnimationFrame(() => render(props.params.params.scene)))

watch(() => props.params.params.scene, render)

onUnmounted(() => {
    handle.value?.dispose()
})
</script>

<template>
    <div style="width: 100%; height: 100%; position: relative">
        <div ref="viewportEl" class="tlmviewer" style="width: 100%; height: 100%"></div>
        <button class="visibility-btn" @click="openVisibilityPanel">Visibility</button>
    </div>
</template>

<style scoped>
.visibility-btn {
    position: absolute;
    bottom: 8px;
    right: 8px;
    z-index: 10;
    background: #1a1a1a;
    border: 1px solid #555;
    color: #ccc;
    font-family: monospace;
    font-size: 11px;
    padding: 3px 10px;
    cursor: pointer;
}

.visibility-btn:hover {
    background: #252525;
    color: #f0f0f0;
    border-color: #888;
}
</style>
