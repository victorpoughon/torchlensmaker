import * as THREE from "three";

import { Line2 } from "three/addons/lines/Line2.js";
import { LineGeometry } from "three/addons/lines/LineGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";

import CustomShaderMaterial from "three-custom-shader-material/vanilla";

import type { SurfaceBaseData } from "tlmprotocol";
import { ElementEventRecord } from "../core/types.ts";
import {
    arrayToMatrix4,
    getTransform2D,
    getTransform3D,
} from "../core/geometry.ts";
import { getRequired } from "../core/utility.ts";

export type { SurfaceBaseData } from "tlmprotocol";

// Parse the fields common to all surfaces from raw JSON
export function parseSurfaceBaseData(raw: any): SurfaceBaseData {
    return {
        type: getRequired<string>(raw, "type"),
        matrix: getRequired<number[][]>(raw, "matrix"),
        clipPlanes: raw.clip_planes ?? [],
    };
}

// Returns a render function suitable for an ElementDescriptor,
// given the two geometry-building functions of a concrete surface.
export function makeSurfaceRender<T extends SurfaceBaseData>(
    makeGeometry2D: (
        data: T,
        tf: THREE.Matrix4,
    ) => [LineGeometry, THREE.Matrix4],
    makeGeometry3D: (
        data: T,
        tf: THREE.Matrix4,
    ) => [THREE.BufferGeometry, THREE.Matrix4, string | null],
): (data: T, dim: number) => THREE.Object3D {
    return (data: T, dim: number): THREE.Object3D => {
        if (dim === 2) {
            return renderSurface2D(data, makeGeometry2D);
        } else {
            return renderSurface3D(data, makeGeometry3D);
        }
    };
}

function renderSurface2D<T extends SurfaceBaseData>(
    data: T,
    makeGeometry2D: (
        data: T,
        tf: THREE.Matrix4,
    ) => [LineGeometry, THREE.Matrix4],
): THREE.Group {
    const group = new THREE.Group();

    const tf_base = getTransform2D(data.matrix);
    const [geometry, transform] = makeGeometry2D(data, tf_base);

    const material = new LineMaterial({
        color: "cyan",
        linewidth: 2,
        worldUnits: false,
        side: THREE.DoubleSide,
    });

    const line_mesh = new Line2(geometry, material);
    line_mesh.applyMatrix4(transform);
    group.add(line_mesh);

    return group;
}

function renderSurface3D<T extends SurfaceBaseData>(
    data: T,
    makeGeometry3D: (
        data: T,
        tf: THREE.Matrix4,
    ) => [THREE.BufferGeometry, THREE.Matrix4, string | null],
): THREE.Group {
    const group = new THREE.Group();

    // Clip planes are encoded as [x, y, z, c], where:
    //     [x, y, z] is the normal vector in surface local frame
    //     c is the constant in surface local frame
    // they must be transformed into world space using the user matrix.
    const userTransform = arrayToMatrix4(data.matrix);
    const clipPlanes: THREE.Plane[] = [];
    for (const plane of data.clipPlanes) {
        const tplane = new THREE.Plane(
            new THREE.Vector3(plane[0], plane[1], plane[2]),
            plane[3],
        );
        tplane.applyMatrix4(userTransform);
        clipPlanes.push(tplane);
    }

    const tf_base = getTransform3D(data.matrix);
    const [geometry, transform, vertexShader] = makeGeometry3D(data, tf_base);

    const material = new CustomShaderMaterial({
        baseMaterial: THREE.MeshNormalMaterial,
        // baseMaterial: THREE.MeshLambertMaterial,
        // baseMaterial: THREE.MeshPhongMaterial,
        vertexShader: vertexShader ?? undefined,

        // Base material properties
        // color: 0x049ef4,
        side: THREE.DoubleSide,
        clippingPlanes: clipPlanes,
        clipIntersection: false,
        transparent: false,
        opacity: 0.8,
        // shininess: 50,
        // specular: 0x5e5e5e,
        // wireframe: true,
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.applyMatrix4(transform);
    group.add(mesh);

    return group;
}

// Returns the default surface event handlers
export function defaultSurfaceEvents<
    T extends SurfaceBaseData,
>(): ElementEventRecord<T> {
    return {
        setCategoryVisibility: (_, object, event) => {
            if (event.category === "surface") {
                object.visible = event.visible;
            }
        },
        setSurfacesColor: (_, object, event) => {
            setColor(object, event.value);
        },
    };
}

function setColor(object: THREE.Object3D, color: THREE.Color): void {
    object.traverse((child) => {
        if (
            child instanceof THREE.Mesh &&
            child.material instanceof THREE.Material &&
            "color" in child.material
        ) {
            (child.material as THREE.Material & { color: THREE.Color }).color =
                color;
        }
    });
}
