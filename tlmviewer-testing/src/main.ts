import tlmviewer from "tlmviewer";
import { builtinScenes } from "./builtinScenes.ts";

async function fetchManifest(): Promise<any> {
    return fetch("/testscenes.json")
        .then((response) => response.json())
        .then((files) => {
            return files;
        });
}

function loadMainViewer(test_file: string) {
    const viewerElement = document.getElementById("main-viewer") as HTMLElement;
    tlmviewer.load(viewerElement, test_file);
}

// Use window.onload to ensure the DOM is fully loaded
window.onload = async () => {
    console.log("loading tlmviewer tests");
    const all_tests = (await fetchManifest()).toSorted();
    console.log(`loaded ${all_tests.length} json test files from manifest`);

    const ul = document.getElementById("tests-list") as HTMLElement;

    // Add built-in test scenes
    for (const { sceneName, data } of builtinScenes) {
        const li = document.createElement("li");
        const a = document.createElement("a");
        a.href = "#";
        a.textContent = `[builtin] ${sceneName}`;
        a.addEventListener("click", (event) => {
            event.preventDefault();
            const viewerElement = document.getElementById(
                "main-viewer",
            ) as HTMLElement;
            tlmviewer.embed(viewerElement, JSON.stringify(data));
        });
        li.appendChild(a);
        ul.appendChild(li);
    }

    // Add json test scenes
    for (const test_file of all_tests) {
        const li = document.createElement("li");
        const a = document.createElement("a");
        a.href = "#";
        a.textContent = test_file;
        a.addEventListener("click", (event) => {
            event.preventDefault();
            loadMainViewer(test_file);
        });
        li.appendChild(a);
        ul.appendChild(li);
    }

    tlmviewer.loadAll();
};
