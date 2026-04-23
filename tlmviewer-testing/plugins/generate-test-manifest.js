import fs from "node:fs/promises";
import { resolve } from "node:path";
import path from "node:path";

// Custom plugin to generate manifest of json test files
export default function testJsonManifestPlugin(options) {
    const { root, tests, manifest } = options;
    const outputFile = resolve(root, "dist/" + manifest);
    const testsDir = resolve(root, tests);

    // Function that
    const generateManifest = async () => {
        const files = await fs.readdir(testsDir);
        const jsonFiles = files.filter(
            (file) => path.extname(file) === ".json"
        );
        const manifest = jsonFiles.map((file) => `/tests/${file}`);

        return JSON.stringify(manifest, null, 2);
    };

    return {
        name: "generate-test-manifest",

        // in build mode, write manifest file
        async writeBundle() {
            const json = await generateManifest();
            await fs.writeFile(outputFile, json);
        },

        // in dev mode, server the manifest route directly
        configureServer(server) {
            server.middlewares.use(async (req, res, next) => {
              if (req.url === '/' + manifest) {
                const json = await generateManifest();
                res.setHeader('Content-Type', 'application/json');
                res.end(json);
              } else {
                next();
              }
            });
          },
    };
}
