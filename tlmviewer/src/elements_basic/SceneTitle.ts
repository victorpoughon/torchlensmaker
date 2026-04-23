import * as THREE from "three";
import type { SceneTitleData } from "tlmprotocol";
import { ElementDescriptor } from "../core/types.ts";
import { getRequired } from "../core/utility.ts";

function parse(raw: any, _dim: number): SceneTitleData {
    return {
        type: "scene-title",
        title: getRequired<string>(raw, "title"),
    };
}

const testData: any[] = [
    {
        type: "scene-title",
        title: "Hello world!",
    },
];

export const sceneTitleDescriptor: ElementDescriptor<SceneTitleData> = {
    type: "scene-title",
    includeInDefaultCamera: false,
    parse,
    render: (_data, _dim) => new THREE.Group(),
    initHTML: (data, _dim, container) => {
        const titleDiv = container.getElementsByClassName("tlmviewer-title")[0];
        titleDiv.textContent = data.title;
    },
    testData2D: testData,
    testData3D: testData,
};
