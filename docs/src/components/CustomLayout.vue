<script setup>
import DefaultTheme from "vitepress/theme";
import { withBase } from "vitepress";

const { Layout } = DefaultTheme;
</script>

<style scoped>
p {
  background: var(--vp-custom-block-warning-bg);
  padding: 1rem;
  border-radius: 10px;
  font-weight: bold;
  font-size: 0.8rem;
  margin: 15px;
}

a {
  font-weight: 500;
  color: var(--vp-c-brand-1);
  text-decoration: underline;
  text-underline-offset: 2px;
  transition: color 0.25s, opacity 0.25s;
}
</style>

<template>
  <Layout>
    <!-- <template #aside-top>
      <p>
        Hello! I'm
        <a class="vp-link" href="https://victorpoughon.fr/">Victor</a>, the
        author of this project.<br />I'm
        <a :href="withBase('/about/#funding')">looking for funding</a> to be
        able to keep working on&nbsp;it. If you can, please consider donating or
        sponsoring.<br />Thank&nbsp;you!&nbsp;❤️
      </p>
    </template> -->
  </Layout>
</template>
