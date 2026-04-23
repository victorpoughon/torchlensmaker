import * as THREE from "three";
import type { Box3DData } from "tlmprotocol";
import { ElementDescriptor } from "../core/types.ts";
import { arrayToMatrix4 } from "../core/geometry.ts";
import { getRequired } from "../core/utility.ts";

function parse(raw: any, _dim: number): Box3DData {
    return {
        type: "box3D",
        size: getRequired<[number, number, number]>(raw, "size"),
        matrix: getRequired<number[][]>(raw, "matrix"),
    };
}

function render(data: Box3DData, _dim: number): THREE.Object3D {
    const { size } = data;

    const group = new THREE.Group();

    const geometry = new THREE.BoxGeometry(
        size[0],
        size[1],
        size[2],
        10,
        10,
        10,
    );
    const material = new THREE.MeshBasicMaterial({
        color: "lightgreen",
        transparent: true,
        opacity: 0.2,
        depthTest: false,
        depthWrite: false,
        wireframe: true,
    });
    const cube = new THREE.Mesh(geometry, material);
    group.add(cube);

    const userTransform = arrayToMatrix4(data.matrix);
    group.applyMatrix4(userTransform);

    return group;
}

const testData3D: any[] = [
    {
        type: "box3D",
        size: [10, 10, 10],
        matrix: [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
    },
    {
        type: "box3D",
        size: [5, 8, 12],
        matrix: [
            [1, 0, 0, 20],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
    },
];

export const box3DDescriptor: ElementDescriptor<Box3DData> = {
    type: "box3D",
    includeInDefaultCamera: true,
    parse,
    render,
    testData2D: [],
    testData3D,
};
