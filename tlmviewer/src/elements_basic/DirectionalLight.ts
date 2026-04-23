import * as THREE from "three";
import type { DirectionalLightData } from "tlmprotocol";
import { ElementDescriptor } from "../core/types.ts";
import { getRequired } from "../core/utility.ts";

function parse(raw: any, _dim: number): DirectionalLightData {
    const position = getRequired<number[]>(raw, "position");
    return {
        type: "directional-light",
        color: getRequired<string>(raw, "color"),
        intensity: getRequired<number>(raw, "intensity"),
        position: position as [number, number, number],
    };
}

function render(data: DirectionalLightData, _dim: number): THREE.Object3D {
    const group = new THREE.Group();
    const light = new THREE.DirectionalLight(data.color, data.intensity);
    light.position.set(data.position[0], data.position[1], data.position[2]);
    group.add(light);
    return group;
}

const testData: any[] = [
    {
        type: "directional-light",
        color: "#ffffff",
        intensity: 0.8,
        position: [1, 2, 3],
    },
];

export const directionalLightDescriptor: ElementDescriptor<DirectionalLightData> =
    {
        type: "directional-light",
        includeInDefaultCamera: false,
        parse,
        render,
        testData2D: testData,
        testData3D: testData,
    };
