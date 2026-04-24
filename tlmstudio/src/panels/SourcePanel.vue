<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { codeToHtml } from 'shiki'
import type { SourceEntry } from '../types.ts'

defineOptions({ name: 'SourcePanel' })

const props = defineProps<{
    params: { params: { source: SourceEntry }; api: unknown; containerApi: unknown }
}>()

const highlighted = ref('')

onMounted(async () => {
    const { filename, language, content } = props.params.params.source
    try {
        highlighted.value = await codeToHtml(content, {
            lang: language,
            theme: 'github-dark',
        })
    } catch {
        // Fall back to plain text if language is unrecognised
        highlighted.value = await codeToHtml(content, {
            lang: 'text',
            theme: 'github-dark',
        })
    }
    // Set document title hint in the panel — the tab title comes from addPanel options
    void filename
})
</script>

<template>
    <div class="source-panel">
        <div class="source-scroll">
            <!-- eslint-disable-next-line vue/no-v-html -->
            <div v-html="highlighted" class="source-code" />
        </div>
    </div>
</template>

<style scoped>
.source-panel {
    height: 100%;
    background: #0d1117;
    overflow: hidden;
    display: flex;
    flex-direction: column;
}

.source-scroll {
    flex: 1;
    overflow: auto;
    padding: 12px 16px;
}

/* Shiki renders a <pre> with inline styles; these override sizing/font */
.source-code :deep(pre) {
    margin: 0;
    padding: 0;
    font-family: monospace;
    font-size: 13px;
    line-height: 1.6;
    background: transparent !important;
    min-width: max-content;
}

.source-code :deep(code) {
    counter-reset: line;
}

.source-code :deep(.line)::before {
    counter-increment: line;
    content: counter(line);
    display: inline-block;
    width: 2em;
    margin-right: 1.5em;
    text-align: right;
    color: #444;
    user-select: none;
}
</style>
