<script setup>

import { ref, onMounted } from 'vue';

const props = defineProps({
    src: String
});

const version = import.meta.env.PACKAGE_VERSION;
const module = await import(`./tlmviewer-${version}.js`);
const tlmviewer = module.default;

var tlmviewerElement = ref(null);

onMounted(() => {
    tlmviewer.load(tlmviewerElement.value, props.src);
});
</script>

<template>
    <div class="tlmviewer" ref="tlmviewerElement"></div>
</template>

<style scoped>
.tlmviewer {
    box-shadow: rgba(0, 0, 0, 0.24) 0px 3px 8px;
    width: calc(100% + 5vw);
    height: 500px;
    background: black;
    margin-top: 1rem;
    margin-bottom: 1rem;

    position: relative;
    left: calc(-5vw / 2);
}

</style>
