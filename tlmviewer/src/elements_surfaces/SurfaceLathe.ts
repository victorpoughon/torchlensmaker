import * as THREE from "three";

import { LineGeometry } from "three/addons/lines/LineGeometry.js";

import { ElementDescriptor } from "../core/types.ts";
import { samples2DToPoints } from "../core/geometry.ts";
import { getRequired } from "../core/utility.ts";

import type { SurfaceLatheData } from "tlmprotocol";
import {
    parseSurfaceBaseData,
    makeSurfaceRender,
    defaultSurfaceEvents,
} from "./surface_utils.ts";

function parse(raw: any, _dim: number): SurfaceLatheData {
    return {
        ...parseSurfaceBaseData(raw),
        type: "surface-lathe",
        samples: getRequired<number[][]>(raw, "samples"),
    };
}

function makeGeometry2D(
    data: SurfaceLatheData,
    tf: THREE.Matrix4,
): [LineGeometry, THREE.Matrix4] {
    const points = samples2DToPoints(data.samples);

    const geometry = new LineGeometry();
    geometry.setPositions(points);

    return [geometry, tf];
}

function makeGeometry3D(
    data: SurfaceLatheData,
    tf: THREE.Matrix4,
): [THREE.BufferGeometry, THREE.Matrix4, string | null] {
    const segments = 50;
    // threejs lathe surface makes a revolution around the Y axis
    // but we want a revolution around the X axis
    // So the procedure is:
    // 1. Swap X/Y coordinates (mirror about the X=Y line)
    // 2. Ask threejs to create the 3D geometry by lathe around the Y axis
    // 3. Swap back by mirroring around the X=Y plane
    // 4. Compose with the matrix4 in the data

    // Make the composed transform
    const flip = new THREE.Matrix4().fromArray([
        0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
    ]);

    const transform = new THREE.Matrix4();
    transform.multiplyMatrices(tf, flip);

    const tpoints = data.samples.map((p) => new THREE.Vector2(p[1], p[0]));

    const geometry = new THREE.LatheGeometry(tpoints, segments);

    return [geometry, transform, null];
}

const testData2D = [
    {
        type: "surface-lathe",
        samples: Array.from({ length: 51 }, (_, i) => {
            const y = -5 + (i * 10) / 50;
            const x = 20 - Math.sign(20) * Math.sqrt(20 * 20 - y * y);
            return [x, y];
        }),
        matrix: [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
    },
    {
        type: "surface-lathe",
        samples: Array.from({ length: 51 }, (_, i) => {
            const y = -5 + (i * 10) / 50;
            const x = -20 - Math.sign(-20) * Math.sqrt(20 * 20 - y * y);
            return [x, y];
        }),
        matrix: [
            [1, 0, 15],
            [0, 1, 0],
            [0, 0, 1],
        ],
    },
];

const testData3D = [
    {
        type: "surface-lathe",
        samples: Array.from({ length: 51 }, (_, i) => {
            const y = (i * 5) / 50;
            const x = 20 - Math.sign(20) * Math.sqrt(20 * 20 - y * y);
            return [x, y];
        }),
        matrix: [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
    },
];

export const surfaceLatheDescriptor: ElementDescriptor<SurfaceLatheData> = {
    type: "surface-lathe",
    includeInDefaultCamera: true,
    parse,
    render: makeSurfaceRender(makeGeometry2D, makeGeometry3D),
    events: defaultSurfaceEvents(),
    testData2D,
    testData3D,
};
