import * as THREE from "three";

import { test, describe } from "node:test";
import assert from "node:assert/strict";
import { allDescriptors } from "../registry.ts";

describe("registry elements parse()", () => {
    for (const descriptor of allDescriptors) {
        for (const data of descriptor.testData2D) {
            test(`[${descriptor.type}] parse(testData2D)`, () => {
                assert.ok(descriptor.parse(data, 2));
            });
        }

        for (const data of descriptor.testData3D) {
            test(`[${descriptor.type}] parse(testData3D)`, () => {
                assert.ok(descriptor.parse(data, 3));
            });
        }
    }
});

describe("registry elements render()", () => {
    for (const descriptor of allDescriptors) {
        for (const raw of descriptor.testData2D) {
            test(`[${descriptor.type}] render(testData2D)`, () => {
                const data = descriptor.parse(raw, 2);
                const obj = descriptor.render(data as any, 2);
                assert.ok(obj instanceof THREE.Object3D);
            });
        }

        for (const raw of descriptor.testData3D) {
            test(`[${descriptor.type}] render(testData3D)`, () => {
                const data = descriptor.parse(raw, 3);
                const obj = descriptor.render(data as any, 3);
                assert.ok(obj instanceof THREE.Object3D);
            });
        }
    }
});

describe("registry elements provide testData2D or testData3D", () => {
    for (const descriptor of allDescriptors) {
        test(`[${descriptor.type}] has test data`, () => {
            assert.ok(
                descriptor.testData2D.length + descriptor.testData3D.length > 0,
                `${descriptor.type} has no test data in either dimension`,
            );
        });
    }
});
