import { resolve } from "path";
import { defineConfig } from "vite";

import cssInjectedByJsPlugin from "vite-plugin-css-injected-by-js";

export default defineConfig({
    resolve: {
        alias: {
            tlmprotocol: resolve(__dirname, "../tlmprotocol/src/index.ts"),
        },
    },
    build: {
        lib: {
            entry: resolve(__dirname, "src/main.ts"),
            name: "tlmviewer",
            fileName: (format) =>
                `tlmviewer-${process.env.npm_package_version}.${format}.js`,
            formats: ["es", "umd"],
        },
    },
    plugins: [
        cssInjectedByJsPlugin(),
    ],
});
