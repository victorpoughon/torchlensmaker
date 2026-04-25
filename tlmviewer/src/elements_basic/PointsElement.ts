import * as THREE from "three";
import type { PointsData } from "tlmprotocol";
import { ElementDescriptor } from "../core/types.ts";
import { getRequired } from "../core/utility.ts";

function parse(raw: any, _dim: number): PointsData {
    return {
        type: "points",
        category: raw.category ?? "",
        vertices: getRequired<number[][]>(raw, "data"),
        color: raw.color ?? "#ffffff",
        radius: raw.radius ?? 0.1,
    };
}

function render(data: PointsData, _dim: number): THREE.Object3D {
    const { vertices, color, radius } = data;
    const count = vertices.length;

    const geometry = new THREE.SphereGeometry(radius, 8, 8);
    const material = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.8 });
    const mesh = new THREE.InstancedMesh(geometry, material, count);

    const dummy = new THREE.Object3D();
    for (let i = 0; i < count; i++) {
        const point = vertices[i];
        dummy.position.set(point[0], point[1], point.length === 2 ? 2.0 : point[2]);
        dummy.updateMatrix();
        mesh.setMatrixAt(i, dummy.matrix);
    }
    mesh.instanceMatrix.needsUpdate = true;

    const group = new THREE.Group();
    group.add(mesh);
    return group;
}

const testData2D = [
    {
        type: "points",
        data: [
            [0, 0],
            [5, 2],
            [10, -1],
            [15, 3],
        ],
    },
];

const testData3D = [
    {
        type: "points",
        data: [
            [0, 0, 0],
            [5, 2, 1],
            [10, -1, 2],
            [15, 3, -1],
        ],
    },
];

export const pointsDescriptor: ElementDescriptor<PointsData> = {
    type: "points",
    includeInDefaultCamera: true,
    parse,
    render,
    events: {
        setCategoryVisibility: (data, object, event) => {
            if (data.category === event.category) {
                object.visible = event.visible;
            }
        },
    },
    testData2D,
    testData3D,
};
