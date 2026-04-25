<script setup lang="ts">
import type { Ref } from 'vue'
import type { SceneEntry } from '../types.ts'
import { formatTime } from '../utils.ts'

defineOptions({ name: 'SceneManagerPanel' })

defineProps<{
    params: {
        params: {
            scenes: Ref<SceneEntry[]>
            openViewport: (scene: SceneEntry) => void
        }
        api: unknown
        containerApi: unknown
    }
}>()
</script>

<template>
    <div class="scene-manager">
        <div class="scene-list">
            <div v-if="params.params.scenes.value.length === 0" class="scene-empty">
                No scenes received yet
            </div>
            <div
                v-for="scene in params.params.scenes.value"
                :key="scene.id"
                class="scene-row"
                @click="params.params.openViewport(scene)"
            >
                <span class="scene-topic">{{ scene.name }}</span>
                <span class="scene-time">{{ formatTime(scene.timestamp) }}</span>
            </div>
        </div>
        <div class="scene-footer">
            <button class="clear-btn" @click="params.params.scenes.value = []">Clear</button>
        </div>
    </div>
</template>

<style scoped>
.scene-manager {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: #0d0d0d;
}

.scene-list {
    flex: 1;
    overflow-y: auto;
    padding: 4px 0;
}

.scene-empty {
    color: #666;
    font-size: 12px;
    padding: 10px 10px;
    font-style: italic;
}

.scene-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 10px;
    font-size: 12px;
    cursor: pointer;
    border-bottom: 1px solid #222;
    gap: 8px;
}

.scene-row:hover {
    background: #1a1a1a;
}

.scene-topic {
    color: #e8e8e8;
    font-weight: bold;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.scene-time {
    color: #888;
    flex-shrink: 0;
    font-size: 11px;
}

.scene-footer {
    border-top: 1px solid #333;
    padding: 6px 10px;
}

.clear-btn {
    background: #1a1a1a;
    border: 1px solid #555;
    color: #ccc;
    font-family: monospace;
    font-size: 11px;
    padding: 3px 10px;
    cursor: pointer;
    width: 100%;
}

.clear-btn:hover {
    background: #252525;
    color: #f0f0f0;
    border-color: #888;
}
</style>
