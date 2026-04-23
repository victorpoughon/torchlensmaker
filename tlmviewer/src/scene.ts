import * as THREE from "three";

import { getRequired } from "./core/utility.ts";

import { defaultSceneElementsData } from "./defaultSceneElements.ts";
import { getMaybeDescriptor } from "./elements_registry/registry.ts";
import { BaseElementData, SceneEntry } from "./core/types.ts";
import { SceneEvent, SceneEventType } from "./core/events.ts";

// Extract available variables from the scene
function extractVariables(root: any): string[] {
    const variables: Set<string> = new Set([]);

    const data = getRequired<any[]>(root, "data");
    for (const element of data) {
        if (getRequired<string>(element, "type") == "rays") {
            const thisVars = element.variables ?? {};
            Object.keys(thisVars).forEach((v: string) => variables.add(v));
        }
    }

    return Array.from(variables);
}

export class TLMScene {
    private root: any;
    private container: HTMLElement;

    // Model
    public sceneGraph: THREE.Group;

    public scene: THREE.Scene;

    public variables: string[];

    constructor(root: any, dim: number, container: HTMLElement) {
        this.root = root;
        this.container = container;
        this.scene = new THREE.Scene();

        this.variables = extractVariables(root);

        // Setup scene graph
        this.sceneGraph = new THREE.Group();
        this.initSceneGraph(dim);

        this.addDefaultSceneElements(dim);

        // Setup the actual THREE scene
        this.scene.add(this.sceneGraph);
    }

    public addSceneElement(dim: number, elementData: any) {
        const descriptor = getMaybeDescriptor(elementData.type);

        // Emit a warning for unknown element types
        if (descriptor === undefined) {
            console.warn(
                `tlmviewer: Unknown scene element ${elementData.type}`,
            );
            return;
        }

        const data: BaseElementData = descriptor.parse(elementData, dim);
        const object3d: THREE.Object3D = descriptor.render(data, dim);
        const entry = new SceneEntry(object3d, data, descriptor);
        object3d.userData = entry;
        this.sceneGraph.add(object3d);

        if (descriptor.initHTML) {
            descriptor.initHTML(data, dim, this.container);
        }
    }

    public addDefaultSceneElements(dim: number) {
        for (const elementData of defaultSceneElementsData(dim)) {
            this.addSceneElement(dim, elementData);
        }
    }

    public initSceneGraph(dim: number) {
        const elements = getRequired<any[]>(this.root, "data");
        for (const elementData of elements) {
            this.addSceneElement(dim, elementData);
        }
    }

    public dispatch<K extends SceneEventType>(event: SceneEvent<K>): void {
        this.sceneGraph.traverse((child: THREE.Object3D) => {
            if (child.userData instanceof SceneEntry) {
                child.userData.onEvent(event);
            }
        });
    }

    // Get bounding box of the scene for the default camera view
    public getBB(): THREE.Box3 {
        const bbox = new THREE.Box3();
        for (const child of this.sceneGraph.children) {
            const element = child.userData;
            const include =
                element instanceof SceneEntry &&
                element.descriptor.includeInDefaultCamera;
            if (include) {
                bbox.union(new THREE.Box3().setFromObject(child));
            }
        }

        if (bbox.isEmpty()) {
            // make sure axes are visible in default empty view
            bbox.min = new THREE.Vector3(-10, -10, -10);
            bbox.max = new THREE.Vector3(10, 10, 10);
        }

        return bbox;
    }
}
