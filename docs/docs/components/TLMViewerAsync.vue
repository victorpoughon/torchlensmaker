<script setup>

import { ref, onMounted } from 'vue';

const props = defineProps({
    src: String
});

console.log("async got prop", props.src);

const module = await import("/tlmviewer.js");
const tlmviewer = module.tlmviewer;

var tlmviewerElement = ref(null);

onMounted(() => {
    const rect = tlmviewerElement.value.getBoundingClientRect();
    console.log("rect", rect);
    tlmviewer.load(tlmviewerElement.value, props.src).then(() => {
        console.log("loaded!");
    });
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
