import * as THREE from "three";
import type { ArrowsData } from "tlmprotocol";
import { ElementDescriptor } from "../core/types.ts";
import { getRequired } from "../core/utility.ts";

function parse(raw: any, dim: number): ArrowsData {
    const arrows = getRequired<number[][]>(raw, "arrows");
    const expectedSize = dim * 2 + 1;
    for (let i = 0; i < arrows.length; i++) {
        if (arrows[i].length !== expectedSize) {
            throw new Error(
                `arrows[${i}] has ${arrows[i].length} elements but expected ${expectedSize} for dim=${dim}`,
            );
        }
    }
    return { type: "arrows", arrows };
}

function render(data: ArrowsData, _dim: number): THREE.Object3D {
    const group = new THREE.Group();

    for (const arrow of data.arrows) {
        let start, end, length;
        if (arrow.length == 5) {
            start = arrow.slice(0, 2);
            end = arrow.slice(2, 4);
            length = arrow[4];
        } else {
            console.assert(arrow.length == 7);
            start = arrow.slice(0, 3);
            end = arrow.slice(3, 6);
            length = arrow[6];
        }

        const dir = new THREE.Vector3(...start);
        dir.normalize();
        const origin = new THREE.Vector3(...end);
        const color = 0xffff00;

        const arrowHelper = new THREE.ArrowHelper(dir, origin, length, color);
        group.add(arrowHelper);
    }

    return group;
}

const testData2D = [
    {
        type: "arrows",
        data: [
            [1, 0, 0, 0, 3],
            [0, 1, 5, 0, 3],
            [1, 1, 10, 0, 3],
        ],
    },
] as unknown as ArrowsData[];

const testData3D = [
    {
        type: "arrows",
        data: [
            [1, 0, 0, 0, 0, 0, 3],
            [0, 1, 0, 5, 0, 0, 3],
            [0, 0, 1, 10, 0, 0, 3],
            [1, 1, 1, 15, 0, 0, 3],
        ],
    },
] as unknown as ArrowsData[];

export const arrowsDescriptor: ElementDescriptor<ArrowsData> = {
    type: "arrows",
    includeInDefaultCamera: true,
    parse,
    render,
    testData2D,
    testData3D,
};
