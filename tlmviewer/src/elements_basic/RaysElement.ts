import * as THREE from "three";
import type { RaysData } from "tlmprotocol";
import { ElementDescriptor } from "../core/types.ts";
import { getRequired } from "../core/utility.ts";

import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";

import { CET_I2, colormap, wavelengthToRgb } from "../color/index.ts";
import { ColorOption } from "../core/events.ts";


function parse(raw: any, dim: number): RaysData {
    return {
        type: "rays",
        category: raw.category ?? "rays-valid",
        points: getRequired<number[][]>(raw, "points"),
        color: raw.color ?? "#ffa724",
        variables: raw.variables ?? {},
        domain: raw.domain ?? {},
        dim: dim as 2 | 3,
    };
}

function makeRays(data: RaysData, colorOption: ColorOption): THREE.Group {
    const { points, color: default_color, variables, domain, dim } = data;
    const expectedLength = 2 * dim;

    const group = new THREE.Group();

    if (!(Symbol.iterator in Object(points))) {
        throw new Error("points field of ray is not iterable");
    }

    // POSITION DATA
    const positions = [];
    for (const ray of points) {
        if (ray.length != expectedLength) {
            throw new Error(
                `Invalid ray array length, got ${ray.length} for dim ${dim}`,
            );
        }

        if (dim == 2) {
            positions.push(ray[0], ray[1], 0, ray[2], ray[3], 0);
        } else {
            positions.push(ray[0], ray[1], ray[2], ray[3], ray[4], ray[5]);
        }
    }

    // COLOR DATA
    const colors = [];
    let use_default_color: boolean = true;

    if (colorOption.colorDim == null) {
        use_default_color = true;
    } else {
        use_default_color = false;
        for (const [index] of points.entries()) {
            let color: Array<number>;

            if (colorOption.trueColor == false) {
                if (!domain.hasOwnProperty(colorOption.colorDim)) {
                    throw new Error(
                        `${colorOption.colorDim} missing from ray domain object`,
                    );
                }
                const [min, max] = domain[colorOption.colorDim];
                const normalizedX = (() => {
                    if (max - min >= 0.001) {
                        return (
                            (variables[colorOption.colorDim][index] - min) /
                            (max - min)
                        );
                    }
                    return 0.5;
                })();
                color = colormap(normalizedX, CET_I2);
            } else {
                const wavelength = variables[colorOption.colorDim][index];
                color = wavelengthToRgb([wavelength])[0];
            }

            const linear_color = new THREE.Color().setRGB(
                color[0],
                color[1],
                color[2],
                THREE.SRGBColorSpace,
            );
            colors.push(...linear_color.toArray(), ...linear_color.toArray());
        }
    }

    const geometry = new LineSegmentsGeometry();
    geometry.setPositions(positions);

    if (!use_default_color) {
        geometry.setColors(colors);
    }

    const material = new LineMaterial({
        ...(use_default_color ? { color: default_color } : {}),
        linewidth: 1,
        vertexColors: !use_default_color,
        dashed: false,
        transparent: true,
    });

    const lines = new LineSegments2(geometry, material);
    group.add(lines);

    return group;
}

function setRaysColorOption(
    object: THREE.Object3D,
    data: RaysData,
    category: string,
    colorOption: ColorOption,
): void {
    if (data.category === category) {
        object.clear();
        object.add(makeRays(data, colorOption));
    }
}

function render(_data: RaysData, _dim: number): THREE.Object3D {
    // Rays are populated reactively via color-option events, not at initial render time
    return new THREE.Group();
}

const testData2D: any[] = [
    {
        type: "rays",
        points: [
            [0, 0, 20, -4],
            [0, 0, 20, -2],
            [0, 0, 20, 0],
            [0, 0, 20, 2],
            [0, 0, 20, 4],
        ],
        color: "#ffa724",
        variables: { field: [-2, -1, 0, 1, 2] },
        domain: { field: [-2, 2] },
    },
];

const testData3D: any[] = [
    {
        type: "rays",
        points: [
            [0, 0, 0, 20, -4, 0],
            [0, 0, 0, 20, -2, 0],
            [0, 0, 0, 20, 0, 0],
            [0, 0, 0, 20, 2, 0],
            [0, 0, 0, 20, 4, 0],
        ],
        color: "#ffa724",
        variables: {},
        domain: {},
    },
];

export const raysDescriptor: ElementDescriptor<RaysData> = {
    type: "rays",
    includeInDefaultCamera: true,
    parse,
    render,
    events: {
        setCategoryVisibility: (data, object, event) => {
            if (data.category === event.category) {
                object.visible = event.visible;
            }
        },
        setValidRaysColor: (data, object, event) => {
            setRaysColorOption(object, data, "rays-valid", event.value);
        },
        setBlockedRaysColor: (data, object, event) => {
            setRaysColorOption(object, data, "rays-blocked", event.value);
        },
        setOutputRaysColor: (data, object, event) => {
            setRaysColorOption(object, data, "rays-output", event.value);
        },
        setRaysOpacity: (_, object, event) => {
            object.traverse((child) => {
                if (
                    child instanceof THREE.Mesh &&
                    child.material instanceof LineMaterial
                ) {
                    child.material.opacity = event.value;
                }
            });
        },
        setRaysThickness: (_, object, event) => {
            object.traverse((child) => {
                if (
                    child instanceof THREE.Mesh &&
                    child.material instanceof LineMaterial
                ) {
                    child.material.linewidth = event.value;
                }
            });
        },
    },
    testData2D,
    testData3D,
};
