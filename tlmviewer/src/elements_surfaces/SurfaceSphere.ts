import * as THREE from "three";

import { LineGeometry } from "three/addons/lines/LineGeometry.js";

import { ElementDescriptor } from "../core/types.ts";
import { samples2DToPoints } from "../core/geometry.ts";
import { getRequired } from "../core/utility.ts";

import type { SurfaceSphereData } from "tlmprotocol";
import {
    parseSurfaceBaseData,
    makeSurfaceRender,
    defaultSurfaceEvents,
} from "./surface_utils.ts";

function parse(raw: any, _dim: number): SurfaceSphereData {
    return {
        ...parseSurfaceBaseData(raw),
        type: "surface-sphere",
        R: getRequired<number>(raw, "R"),
    };
}

function makeGeometry2D(
    data: SurfaceSphereData,
    tf: THREE.Matrix4,
): [LineGeometry, THREE.Matrix4] {
    const { R } = data;
    const N = 101;
    const samples: Array<[number, number]> = Array.from({ length: N }, (_, i) => {
        const t = (2 * Math.PI * i) / (N - 1);
        return [R * Math.cos(t), R * Math.sin(t)];
    });
    const points = samples2DToPoints(samples);

    const geometry = new LineGeometry();
    geometry.setPositions(points);

    return [geometry, tf];
}

function makeGeometry3D(
    data: SurfaceSphereData,
    tf: THREE.Matrix4,
): [THREE.BufferGeometry, THREE.Matrix4, string | null] {
    const geometry = new THREE.SphereGeometry(Math.abs(data.R), 32, 16);
    return [geometry, tf, null];
}

const testData2D = [
    {
        type: "surface-sphere",
        R: 5,
        matrix: [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
    },
    {
        type: "surface-sphere",
        R: 5,
        matrix: [
            [1, 0, 15],
            [0, 1, 0],
            [0, 0, 1],
        ],
    },
];

const testData3D = [
    {
        type: "surface-sphere",
        R: 5,
        matrix: [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
    },
    {
        type: "surface-sphere",
        R: 5,
        matrix: [
            [1, 0, 0, 15],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
    },
];

export const surfaceSphereDescriptor: ElementDescriptor<SurfaceSphereData> = {
    type: "surface-sphere",
    includeInDefaultCamera: true,
    parse,
    render: makeSurfaceRender(makeGeometry2D, makeGeometry3D),
    events: defaultSurfaceEvents(),
    testData2D,
    testData3D,
};
