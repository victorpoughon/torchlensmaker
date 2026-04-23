import * as THREE from "three";

import { LineGeometry } from "three/addons/lines/LineGeometry.js";

import { ElementDescriptor } from "../core/types.ts";
import { getRequired } from "../core/utility.ts";

import type { SurfaceSagData } from "tlmprotocol";
import {
    parseSurfaceBaseData,
    makeSurfaceRender,
    defaultSurfaceEvents,
} from "./surface_utils.ts";
import { parseSagFunction, glslRender } from "./sagFunctions.ts";

function parse(raw: any, _dim: number): SurfaceSagData {
    return {
        ...parseSurfaceBaseData(raw),
        type: "surface-sag",
        diameter: getRequired<number>(raw, "diameter"),
        sagFunctionData: getRequired<any>(raw, "sag-function"),
    };
}

// Generate line geometry by sampling a sag function in 2D
function sagGeometry2D(
    sag: (r: number) => number,
    start: number,
    end: number,
    N: number,
    const_z: number,
): LineGeometry {
    const geometry = new LineGeometry();

    const step = (end - start) / (N - 1);
    const points: number[] = [];

    for (let i = 0; i < N; i++) {
        const y = start + i * step;
        const x = sag(y);

        points.push(x, y, const_z);
    }

    geometry.setPositions(points);
    return geometry;
}

function makeGeometry2D(
    data: SurfaceSagData,
    tf: THREE.Matrix4,
): [LineGeometry, THREE.Matrix4] {
    const { diameter } = data;
    const tau = diameter / 2;
    const sag = parseSagFunction(data.sagFunctionData as any, tau).sagFunction2D(tau);

    const geometry = sagGeometry2D(sag, -diameter / 2, diameter / 2, 100, 1.0);

    return [geometry, tf];
}

function makeGeometry3D(
    data: SurfaceSagData,
    tf: THREE.Matrix4,
): [THREE.BufferGeometry, THREE.Matrix4, string | null] {
    // We use ring geometry as the base geometry
    // But could consider using a custom geometry
    // for better distribution of vertices over the disk
    const { diameter } = data;
    const geometry = new THREE.RingGeometry(0, diameter / 2, 256, 256).rotateY(
        Math.PI / 2,
    );

    const tau = diameter / 2;
    const sag = parseSagFunction(data.sagFunctionData as any, tau);
    const vertexShader = glslRender(
        sag.shaderG(tau),
        sag.shaderGgrad(tau),
        sag.name,
    );

    return [geometry, tf, vertexShader];
}

const testData2D = [
    {
        type: "surface-sag",
        diameter: 10,
        "sag-function": { "sag-type": "spherical", C: 0.1 },
        matrix: [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
    },
    {
        type: "surface-sag",
        diameter: 10,
        "sag-function": { "sag-type": "parabolic", A: -0.05 },
        matrix: [
            [1, 0, 15],
            [0, 1, 0],
            [0, 0, 1],
        ],
    },
    {
        type: "surface-sag",
        diameter: 10,
        "sag-function": { "sag-type": "conical", C: 0.1, K: -1.5 },
        matrix: [
            [1, 0, 30],
            [0, 1, 0],
            [0, 0, 1],
        ],
    },
    {
        type: "surface-sag",
        diameter: 10,
        "sag-function": {
            "sag-type": "aspheric",
            C: 0.1,
            K: 0,
            coefficients: [0, -1e-4, 0],
        },
        matrix: [
            [1, 0, 45],
            [0, 1, 0],
            [0, 0, 1],
        ],
    },
    {
        type: "surface-sag",
        diameter: 10,
        "sag-function": {
            "sag-type": "sum",
            terms: [
                { "sag-type": "spherical", C: 0.08 },
                { "sag-type": "parabolic", A: 0.02 },
            ],
        },
        matrix: [
            [1, 0, 60],
            [0, 1, 0],
            [0, 0, 1],
        ],
    },
];

const testData3D = [
    {
        type: "surface-sag",
        diameter: 10,
        "sag-function": { "sag-type": "spherical", C: 0.1 },
        matrix: [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
    },
    {
        type: "surface-sag",
        diameter: 10,
        "sag-function": { "sag-type": "parabolic", A: -0.05 },
        matrix: [
            [1, 0, 0, 15],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
    },
    {
        type: "surface-sag",
        diameter: 10,
        "sag-function": { "sag-type": "conical", C: 0.1, K: -1.5 },
        matrix: [
            [1, 0, 0, 30],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
    },
    {
        type: "surface-sag",
        diameter: 10,
        "sag-function": {
            "sag-type": "aspheric",
            C: 0.1,
            K: 0,
            coefficients: [5e-3, -3e-5, 0],
        },
        matrix: [
            [1, 0, 0, 45],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
    },
    {
        type: "surface-sag",
        diameter: 10,
        "sag-function": {
            "sag-type": "xypolynomial",
            coefficients: [
                [0, 0.05, 0],
                [0.05, 0, 0],
                [0.01, 0, -0.01],
            ],
        },
        matrix: [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
    },
];

export const surfaceSagDescriptor: ElementDescriptor<SurfaceSagData> = {
    type: "surface-sag",
    includeInDefaultCamera: true,
    parse,
    render: makeSurfaceRender(makeGeometry2D, makeGeometry3D),
    events: defaultSurfaceEvents(),
    testData2D,
    testData3D,
};
