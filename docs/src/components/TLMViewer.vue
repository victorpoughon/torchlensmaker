<script setup>
const props = defineProps({
    src: String
});
</script>

<template>
    <Suspense>
        <TLMViewerAsync :src="src"/>
    </Suspense>
</template>
