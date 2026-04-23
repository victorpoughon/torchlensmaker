import * as THREE from "three";
import type { CylinderData } from "tlmprotocol";
import { ElementDescriptor } from "../core/types.ts";
import { arrayToMatrix4, homogeneousMatrix3to4 } from "../core/geometry.ts";
import { getRequired } from "../core/utility.ts";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";

function parse(raw: any, _dim: number): CylinderData {
    return {
        type: "cylinder",
        matrix: getRequired<number[][]>(raw, "matrix"),
        xmin: getRequired<number>(raw, "xmin"),
        xmax: getRequired<number>(raw, "xmax"),
        radius: getRequired<number>(raw, "radius"),
    };
}

// Create a circle centered at (x, 0, 0) in the YZ plane
function circleGeometry(x: number, radius: number): LineSegmentsGeometry {
    const geometry = new LineSegmentsGeometry();
    const positions: number[] = [];
    const segments = 64;

    for (let i = 0; i < segments; i++) {
        const theta1 = (i / segments) * Math.PI * 2;
        const theta2 = ((i + 1) / segments) * Math.PI * 2;

        const y1 = radius * Math.cos(theta1);
        const z1 = radius * Math.sin(theta1);
        const y2 = radius * Math.cos(theta2);
        const z2 = radius * Math.sin(theta2);

        positions.push(0, y1, z1, 0, y2, z2);
    }

    geometry.setPositions(positions);
    geometry.translate(x, 0, 0);
    return geometry;
}

function render(data: CylinderData, _dim: number): THREE.Object3D {
    const { matrix, xmin, xmax, radius } = data;

    const group = new THREE.Group();

    const lineMaterial = new LineMaterial({
        linewidth: 2,
        color: "lightgreen",
        dashed: false,
        transparent: false,
    });

    const surfaceMaterial = new THREE.MeshBasicMaterial({
        color: "lightgreen",
        transparent: true,
        opacity: 0.2,
        depthTest: false,
        depthWrite: false,
    });

    // Detect 2D (3x3 matrix) vs 3D (4x4 matrix) by matrix size
    if (matrix.length === 3) {
        const positions: number[] = [];
        const z = 0.0;
        positions.push(xmin, -radius, z, xmin, radius, z);
        positions.push(xmin, radius, z, xmax, radius, z);
        positions.push(xmax, radius, z, xmax, -radius, z);
        positions.push(xmax, -radius, z, xmin, -radius, z);

        const geometry = new LineSegmentsGeometry();
        geometry.setPositions(positions);
        group.add(new LineSegments2(geometry, lineMaterial));

        const userTransform = arrayToMatrix4(homogeneousMatrix3to4(matrix));
        group.applyMatrix4(userTransform);
    } else {
        const height = xmax - xmin;
        const center = xmin + height / 2;

        const cylinder = new THREE.CylinderGeometry(radius, radius, height, 64);
        cylinder.rotateZ(Math.PI / 2);
        cylinder.translate(center, 0.0, 0.0);

        group.add(
            new LineSegments2(circleGeometry(xmin, radius), lineMaterial),
        );
        group.add(
            new LineSegments2(circleGeometry(xmax, radius), lineMaterial),
        );
        group.add(new THREE.Mesh(cylinder, surfaceMaterial));

        const userTransform = arrayToMatrix4(matrix);
        group.applyMatrix4(userTransform);
    }

    return group;
}

const testData2D = [
    {
        type: "cylinder",
        xmin: -1,
        xmax: 1.27,
        radius: 5,
        matrix: [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
    },
];

const testData3D = [
    {
        type: "cylinder",
        xmin: -10,
        xmax: 1.27,
        radius: 5,
        matrix: [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
    },
];

export const cylinderDescriptor: ElementDescriptor<CylinderData> = {
    type: "cylinder",
    includeInDefaultCamera: true,
    parse,
    render,
    events: {
        setCategoryVisibility: (_, object, event) => {
            if (event.category === "bcyl") {
                object.visible = event.visible;
            }
        },
    },
    testData2D,
    testData3D,
};
