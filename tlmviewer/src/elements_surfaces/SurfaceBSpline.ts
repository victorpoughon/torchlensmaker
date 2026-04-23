import * as THREE from "three";

import { LineGeometry } from "three/addons/lines/LineGeometry.js";
import * as NURBSUtils from "three/addons/curves/NURBSUtils.js";
import { ParametricGeometry } from "three/addons/geometries/ParametricGeometry.js";

import { ElementDescriptor } from "../core/types.ts";
import { getRequired } from "../core/utility.ts";

import type { SurfaceBSplineData } from "tlmprotocol";
import {
    parseSurfaceBaseData,
    makeSurfaceRender,
    defaultSurfaceEvents,
} from "./surface_utils.ts";

function linspace(start: number, stop: number, num: number): number[] {
    if (num === 0) return [];
    if (num === 1) return [start];
    return Array.from(
        { length: num },
        (_, i) => start + (i / (num - 1)) * (stop - start),
    );
}

// Generate a uniform knot vector for K control points with degree p.
//
// Inner knots are K-p+1 values uniformly spaced in [0, 1] (includes 0 and 1).
//
// clamped:   p zeros prepended, p ones appended — surface interpolates endpoints
// unclamped: p leading knots extending below 0, p trailing knots extending above 1
//
// Matches the Python reference: all_knots(K, P, clamped=...) with M = K - P.
export function uniformKnots(
    K: number,
    p: number,
    knotType: "clamped" | "unclamped",
): number[] {
    const inner = linspace(0, 1, K - p + 1);
    if (knotType === "clamped") {
        return [
            ...Array<number>(p).fill(0),
            ...inner,
            ...Array<number>(p).fill(1),
        ];
    } else {
        const M = K - p;
        const leading = linspace(-p / M, -1 / M, p);
        const trailing = linspace((1 + M) / M, 1 + p / M, p);
        return [...leading, ...inner, ...trailing];
    }
}

function parse(raw: any, _dim: number): SurfaceBSplineData {
    const points: number[][][] = getRequired(raw, "points");
    const nrows = points.length;
    const ncols = points[0]?.length ?? 0;
    const weights: number[][] =
        raw.weights ??
        Array.from({ length: nrows }, () => Array<number>(ncols).fill(1.0));
    return {
        ...parseSurfaceBaseData(raw),
        type: "surface-bspline",
        points,
        weights,
        degree: getRequired(raw, "degree"),
        knotType: getRequired(raw, "knot_type"),
        samples: raw.samples ?? [64, 64],
    };
}

function makeGeometry2D(
    _data: SurfaceBSplineData,
    _tf: THREE.Matrix4,
): [LineGeometry, THREE.Matrix4] {
    throw new Error("surface-bspline does not support 2D rendering");
}

function makeGeometry3D(
    data: SurfaceBSplineData,
    tf: THREE.Matrix4,
): [THREE.BufferGeometry, THREE.Matrix4, null] {
    const [degU, degV] = data.degree;
    const nrows = data.points.length;
    const ncols = data.points[0].length;

    const knotsU = uniformKnots(nrows, degU, data.knotType);
    const knotsV = uniformKnots(ncols, degV, data.knotType);

    const controlPoints: THREE.Vector4[][] = data.points.map((row, i) =>
        row.map((pt, j) => {
            const w = data.weights[i][j];
            return new THREE.Vector4(pt[0] * w, pt[1] * w, pt[2] * w, w);
        }),
    );

    const [slices, stacks] = data.samples;
    // NURBSSurface.getPoint maps t∈[0,1] to [knots[0], knots[last]], which is
    // wrong for unclamped knots (guard knots extend outside [0,1]).
    // Our knot vectors always have valid domain [U[p], U[n]] = [0, 1], so we
    // pass t directly as u/v to calcSurfacePoint.
    const geometry = new ParametricGeometry(
        (u: number, v: number, target: THREE.Vector3) =>
            NURBSUtils.calcSurfacePoint(
                degU,
                degV,
                knotsU,
                knotsV,
                controlPoints,
                u,
                v,
                target,
            ),
        slices,
        stacks,
    );

    return [geometry, tf, null];
}

const testData3D = [
    {
        type: "surface-bspline",
        degree: [2, 2],
        knot_type: "unclamped",
        points: [
            [
                [2.526900053024292, -2.052299976348877, -0.4562999904155731],
                [2.6080000400543213, -1.7411999702453613, 0.44679999351501465],
                [1.7403000593185425, -1.2918000221252441, 0.6848999857902527],
                [1.2460999488830566, -0.9594600200653076, 0.008999999612569809],
                [1.7727999687194824, -1.1412999629974365, -0.6757000088691711],
                [2.526900053024292, -2.052299976348877, -0.4562999904155731],
                [2.6080000400543213, -1.7411999702453613, 0.44679999351501465],
            ],
            [
                [2.60479998588562, 1.9523999691009521, -0.42739999294281006],
                [2.538599967956543, 1.8327000141143799, 0.4311000108718872],
                [1.7462999820709229, 1.2472000122070312, 0.6771000027656555],
                [1.2805999517440796, 0.9645100235939026, 0.020600000396370888],
                [1.739300012588501, 1.1967999935150146, -0.7109000086784363],
                [2.60479998588562, 1.9523999691009521, -0.42739999294281006],
                [2.538599967956543, 1.8327000141143799, 0.4311000108718872],
            ],
            [
                [-0.9908000230789185, 3.08870005607605, -0.4426000118255615],
                [-0.9763000011444092, 2.913800001144409, 0.4357999861240387],
                [-0.6586999893188477, 2.1363000869750977, 0.7107999920845032],
                [
                    -0.5077999830245972, 1.4337999820709229,
                    -0.011099999770522118,
                ],
                [-0.6409000158309937, 2.047100067138672, -0.6759999990463257],
                [-0.9908000230789185, 3.08870005607605, -0.4426000118255615],
                [-0.9763000011444092, 2.913800001144409, 0.4357999861240387],
            ],
            [
                [
                    -3.1939001083374023, -0.1458600014448166,
                    -0.43309998512268066,
                ],
                [-3.1301000118255615, 0.11984000355005264, 0.40869998931884766],
                [-2.199899911880493, -0.037429001182317734, 0.6898000240325928],
                [
                    -1.5038000345230103, 0.0009815000230446458,
                    0.004100000020116568,
                ],
                [-2.1882998943328857, 0.0771000012755394, -0.7149999737739563],
                [
                    -3.1939001083374023, -0.1458600014448166,
                    -0.43309998512268066,
                ],
                [-3.1301000118255615, 0.11984000355005264, 0.40869998931884766],
            ],
            [
                [-0.9646000266075134, -2.734100103378296, -0.4278999865055084],
                [-1.0085999965667725, -3.200200080871582, 0.4325000047683716],
                [-0.6402999758720398, -1.9943000078201294, 0.7091000080108643],
                [
                    -0.5058000087738037, -1.4261000156402588,
                    -0.00800000037997961,
                ],
                [-0.6628000140190125, -2.2314999103546143, -0.6866000294685364],
                [-0.9646000266075134, -2.734100103378296, -0.4278999865055084],
                [-1.0085999965667725, -3.200200080871582, 0.4325000047683716],
            ],
            [
                [2.526900053024292, -2.052299976348877, -0.4562999904155731],
                [2.6080000400543213, -1.7411999702453613, 0.44679999351501465],
                [1.7403000593185425, -1.2918000221252441, 0.6848999857902527],
                [1.2460999488830566, -0.9594600200653076, 0.008999999612569809],
                [1.7727999687194824, -1.1412999629974365, -0.6757000088691711],
                [2.526900053024292, -2.052299976348877, -0.4562999904155731],
                [2.6080000400543213, -1.7411999702453613, 0.44679999351501465],
            ],
            [
                [2.60479998588562, 1.9523999691009521, -0.42739999294281006],
                [2.538599967956543, 1.8327000141143799, 0.4311000108718872],
                [1.7462999820709229, 1.2472000122070312, 0.6771000027656555],
                [1.2805999517440796, 0.9645100235939026, 0.020600000396370888],
                [1.739300012588501, 1.1967999935150146, -0.7109000086784363],
                [2.60479998588562, 1.9523999691009521, -0.42739999294281006],
                [2.538599967956543, 1.8327000141143799, 0.4311000108718872],
            ],
        ],
        matrix: [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
    },
    {
        type: "surface-bspline",
        degree: [3, 3],
        knot_type: "clamped",
        samples: [32, 32],
        points: [
            [
                [-5, -5, 0],
                [-5, -1, 0],
                [-5, 1, 0],
                [-5, 5, 0],
            ],
            [
                [-1, -5, 0],
                [-1, -1, 3],
                [-1, 1, 3],
                [-1, 5, 0],
            ],
            [
                [1, -5, 0],
                [1, -1, 3],
                [1, 1, 3],
                [1, 5, 0],
            ],
            [
                [5, -5, 0],
                [5, -1, 0],
                [5, 1, 0],
                [5, 5, 0],
            ],
        ],
        matrix: [
            [1, 0, 0, 15],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
    },
];

export const surfaceBSplineDescriptor: ElementDescriptor<SurfaceBSplineData> = {
    type: "surface-bspline",
    includeInDefaultCamera: true,
    parse,
    render: makeSurfaceRender(makeGeometry2D, makeGeometry3D),
    events: defaultSurfaceEvents(),
    testData2D: [],
    testData3D,
};
