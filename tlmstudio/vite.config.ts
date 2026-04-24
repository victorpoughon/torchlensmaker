import { fileURLToPath, URL } from 'node:url'
import { resolve } from 'node:path'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import vueDevTools from 'vite-plugin-vue-devtools'

// https://vite.dev/config/
export default defineConfig({
    plugins: [vue(), vueDevTools()],
    resolve: {
        alias: {
            '@': fileURLToPath(new URL('./src', import.meta.url)),
            tlmprotocol: resolve(__dirname, '../tlmprotocol/src/index.ts'),
        },
    },
    server: {
        proxy: {
            '/ws': {
                target: 'ws://localhost:8765',
                ws: true,
            },
        },
    },
})
