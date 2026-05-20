<script setup lang="ts">
import { ref, watch } from 'vue'
import type { Ref } from 'vue'
import type { RenderHandle } from '../../../tlmviewer/src/render.ts'
import type { SceneElementInfo } from '../../../tlmviewer/src/core/types.ts';

defineOptions({ name: 'VisibilityPanel' })

const props = defineProps<{
    params: {
        params: { handle: Ref<RenderHandle | null> }
        api: unknown
        containerApi: unknown
    }
}>()

const TYPE_LABELS: Record<string, string> = {
    'ambient-light':    'Ambient Light',
    'arrows':           'Arrows',
    'box3D':            'Box',
    'cylinder':         'Cylinder',
    'directional-light':'Directional Light',
    'points':           'Points',
    'rays':             'Rays',
    'scene-axis':       'Axis',
    'scene-title':      'Title',
    'surface-bspline':  'Surface (B-Spline)',
    'surface-disk':     'Surface (Disk)',
    'surface-lathe':    'Surface (Lathe)',
    'surface-sag':      'Surface (Sag)',
    'surface-sphere':   'Surface (Sphere)',
    'surface-sphere-r': 'Surface (Sphere)',
}

function typeLabel(type: string): string {
    return TYPE_LABELS[type] ?? type
}

const elements = ref<SceneElementInfo[]>([])

watch(
    () => props.params.params.handle.value,
    (h) => { elements.value = h ? h.getElements() : [] },
    { immediate: true },
)

function toggle(el: SceneElementInfo) {
    const next = !el.visible
    props.params.params.handle.value?.setElementVisibility(el.id, next)
    el.visible = next
}
</script>

<template>
    <div class="visibility-panel">
        <div v-if="elements.length === 0" class="empty">No scene loaded</div>
        <div
            v-for="el in elements"
            :key="el.id"
            class="row"
            @click="toggle(el)"
        >
            <input type="checkbox" :checked="el.visible" @click.stop="toggle(el)" />
            <span class="label">{{ typeLabel(el.type) }}</span>
            <span v-if="el.category" class="category">{{ el.category }}</span>
        </div>
    </div>
</template>

<style scoped>
.visibility-panel {
    height: 100%;
    overflow-y: auto;
    background: #0d0d0d;
    padding: 4px 0;
}

.empty {
    color: #666;
    font-size: 12px;
    padding: 10px;
    font-style: italic;
}

.row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 10px;
    font-size: 12px;
    cursor: pointer;
    border-bottom: 1px solid #1a1a1a;
    font-family: monospace;
}

.row:hover {
    background: #1a1a1a;
}

.label {
    color: #e8e8e8;
    flex: 1;
}

.category {
    color: #888;
    font-size: 11px;
    flex-shrink: 0;
}
</style>
