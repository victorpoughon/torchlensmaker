import * as THREE from "three";
import type { SceneAxisData } from "tlmprotocol";
import { ElementDescriptor } from "../core/types.ts";
import { getOption, getRequired } from "../core/utility.ts";
import { makeLine2 } from "../core/lineUtils.ts";

function parse(raw: any, _dim: number): SceneAxisData {
    return {
        type: "scene-axis",
        axis: getOption(raw, "axis", ["x", "y", "z"]),
        length: getRequired<number>(raw, "length"),
        color: raw["color"] ?? "#e3e3e3",
    };
}

function render(data: SceneAxisData, _dim: number): THREE.Object3D {
    const group = new THREE.Group();

    const { axis, length, color } = data;

    if (axis == "x") {
        group.add(makeLine2([-length, 0, 0], [length, 0, 0], color));
    } else if (axis == "y") {
        group.add(makeLine2([0, -length, 0], [0, length, 0], color));
    } else if (axis == "z") {
        group.add(makeLine2([0, 0, -length], [0, 0, length], color));
    }

    return group;
}

const testData: any[] = [
    {
        type: "scene-axis",
        axis: "x",
        length: 10,
        color: "#e3e3e3",
    },
];

export const sceneAxisDescriptor: ElementDescriptor<SceneAxisData> = {
    type: "scene-axis",
    includeInDefaultCamera: false,
    parse,
    render,
    events: {
        setCategoryVisibility: (data, object, event) => {
            if (event.category === "axis-x" && data.axis === "x") {
                object.visible = event.visible;
            } else if (event.category === "axis-y" && data.axis === "y") {
                object.visible = event.visible;
            } else if (event.category === "axis-z" && data.axis === "z") {
                object.visible = event.visible;
            }
        },
    },
    testData2D: testData,
    testData3D: testData,
};
