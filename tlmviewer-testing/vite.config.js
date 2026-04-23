import { defineConfig } from "vite";
import path from "node:path";
import testJsonManifestPlugin from "./plugins/generate-test-manifest.js";

export default defineConfig({
    resolve: {
        alias: {
            tlmviewer: path.resolve(__dirname, "../tlmviewer/src/main.ts"),
            tlmprotocol: path.resolve(__dirname, "../tlmprotocol/src/index.ts"),
        },
    },
    plugins: [
        testJsonManifestPlugin({ root: __dirname, tests: "public/tests", manifest: "testscenes.json" }),
    ],
});
