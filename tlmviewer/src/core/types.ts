import * as THREE from "three";
import { SceneEventType, SceneEvent } from "./events.ts";

import type { BaseElementData } from "tlmprotocol";
export type { BaseElementData };

// Handler for a specific event type
export type EventHandler<
    T extends BaseElementData,
    K extends SceneEventType,
> = (data: T, object: THREE.Object3D, event: SceneEvent<K>) => void;

// The record an element provides: a subset of event types it cares about
export type ElementEventRecord<T extends BaseElementData> = {
    [K in SceneEventType]?: EventHandler<T, K>;
};

export type ElementDescriptor<T extends BaseElementData> = {
    type: T["type"];
    includeInDefaultCamera: boolean;
    parse: (raw: unknown, dim: number) => T;
    render: (data: T, dim: number) => THREE.Object3D;
    initHTML?: (data: T, dim: number, container: Element) => void;
    events?: ElementEventRecord<T>;
    testData2D: any[];
    testData3D: any[];
};

// This object type is used as the user data of the three js object
export class SceneEntry {
    public object: THREE.Object3D;
    readonly data: BaseElementData;
    readonly descriptor: ElementDescriptor<BaseElementData>;

    constructor(
        object: THREE.Object3D,
        data: BaseElementData,
        descriptor: ElementDescriptor<BaseElementData>,
    ) {
        this.object = object;
        this.data = data;
        this.descriptor = descriptor;
    }

    public onEvent<K extends SceneEventType>(event: SceneEvent<K>): void {
        if (
            this.descriptor.events === undefined ||
            this.descriptor.events === null
        )
            return;

        const handler = this.descriptor.events[event.type];
        if (handler) {
            handler(this.data, this.object, event);
        }
    }
}
