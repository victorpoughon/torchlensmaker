import { version } from "../package.json";
import { allDescriptors } from "./elements_registry/registry.ts";
import { parseSagFunction, glslRender } from "./elements_surfaces/sagFunctions.ts";
import { renderScene } from "./render.ts";
import { connect } from "./connect.ts";
import "./viewer.css";

function embed(container: HTMLElement, json_data: string) {
    try {
        const data = JSON.parse(json_data);
        renderScene(container, data);
    } catch (error) {
        container.innerHTML =
            "<span style='color: red'>tlmviewer error: " + error + "</span>";
        throw error;
    }
}

async function load(container: HTMLElement, url: string): Promise<void> {
    try {
        const response = await fetch(url);
        const data = await response.json();
        renderScene(container, data);
    } catch (error) {
        container.innerHTML =
            "<span style='color: red'>tlmviewer error: " + error + "</span>";
        throw error;
    }
}

async function loadAll(): Promise<Promise<void>[]> {
    const elements = document.querySelectorAll(".tlmviewer");
    const promises: Promise<void>[] = [];

    elements.forEach((element) => {
        const url = element.getAttribute("data-url");
        if (url) {
            promises.push(load(element as HTMLElement, url));
        }
    });

    return promises;
}

export default {
    embed,
    load,
    loadAll,
    connect,
    testing: {
        allDescriptors,
        parseSagFunction,
        glslRender,
    },
};

console.log(`tlmviewer-${version} loaded`);
