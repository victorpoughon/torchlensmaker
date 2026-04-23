import * as THREE from "three";

import { LineGeometry } from "three/addons/lines/LineGeometry.js";

import { ElementDescriptor } from "../core/types.ts";
import { samples2DToPoints } from "../core/geometry.ts";
import { getRequired } from "../core/utility.ts";

import type { SurfaceSphereRData } from "tlmprotocol";
import {
    parseSurfaceBaseData,
    makeSurfaceRender,
    defaultSurfaceEvents,
} from "./surface_utils.ts";

function parse(raw: any, _dim: number): SurfaceSphereRData {
    return {
        ...parseSurfaceBaseData(raw),
        type: "surface-sphere-r",
        R: getRequired<number>(raw, "R"),
        diameter: getRequired<number>(raw, "diameter"),
    };
}

/**
 * Angular sampling of a circular arc defined by radius.
 * Returns an array of [x, y] coordinates.
 */
function sphereSamplesAngular2(
    radius: number,
    start: number,
    end: number,
    N: number,
): Array<[number, number]> {
    // Generate theta values based on the radius sign
    const theta =
        radius > 0
            ? Array.from(
                  { length: N },
                  (_, i) => Math.PI - end + (i * (end - start)) / (N - 1),
              )
            : Array.from(
                  { length: N },
                  (_, i) => start + (i * (end - start)) / (N - 1),
              );

    // Compute X and Y coordinates
    const X = theta.map((t) => Math.abs(radius) * Math.cos(t) + radius);
    const Y = theta.map((t) => Math.abs(radius) * Math.sin(t));

    // Combine X and Y into pairs
    return X.map((x, i) => [x, Y[i]]);
}

function makeGeometry2D(
    data: SurfaceSphereRData,
    tf: THREE.Matrix4,
): [LineGeometry, THREE.Matrix4] {
    const { R: arc_radius, diameter } = data;
    const thetaMax = Math.asin(diameter / 2 / Math.abs(arc_radius));
    const samples = sphereSamplesAngular2(arc_radius, -thetaMax, thetaMax, 101);
    const points = samples2DToPoints(samples);

    const geometry = new LineGeometry();
    geometry.setPositions(points);

    return [geometry, tf];
}

function makeGeometry3D(
    data: SurfaceSphereRData,
    tf: THREE.Matrix4,
): [THREE.BufferGeometry, THREE.Matrix4, string | null] {
    const { R: arc_radius, diameter } = data;
    const thetaMax = Math.asin(diameter / 2 / Math.abs(arc_radius));
    const samples = sphereSamplesAngular2(arc_radius, 0.0, thetaMax, 101);

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

    const tpoints = samples.map((p) => new THREE.Vector2(p[1], p[0]));

    const geometry = new THREE.LatheGeometry(tpoints, segments);

    return [geometry, transform, null];
}

const testData2D = [
    {
        type: "surface-sphere-r",
        diameter: 10,
        R: 20,
        matrix: [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
    },
    {
        type: "surface-sphere-r",
        diameter: 10,
        R: -20,
        matrix: [
            [1, 0, 15],
            [0, 1, 0],
            [0, 0, 1],
        ],
    },
];

const testData3D = [
    {
        type: "surface-sphere-r",
        diameter: 10,
        R: 20,
        matrix: [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
    },
    {
        type: "surface-sphere-r",
        diameter: 10,
        R: -20,
        matrix: [
            [1, 0, 0, 15],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
    },
];

export const surfaceSphereRDescriptor: ElementDescriptor<SurfaceSphereRData> = {
    type: "surface-sphere-r",
    includeInDefaultCamera: true,
    parse,
    render: makeSurfaceRender(makeGeometry2D, makeGeometry3D),
    events: defaultSurfaceEvents(),
    testData2D,
    testData3D,
};
