import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type { CameraRig, CameraState } from "./CameraRig.ts";

export function createCamera2D(domElement: HTMLElement): CameraRig {
    const camera = new THREE.OrthographicCamera(-10, 10, 10, -10, -10000, 10000);

    const controls = new OrbitControls(camera, domElement);
    controls.enableRotate = false;
    controls.mouseButtons = {
        LEFT: THREE.MOUSE.PAN,
        RIGHT: THREE.MOUSE.PAN,
    };

    return {
        camera,
        controls,

        fitToBox(bbox: THREE.Box3, aspect: number): void {
            const center = new THREE.Vector3();
            bbox.getCenter(center);
            const size = new THREE.Vector3();
            bbox.getSize(size);
            const marginFactor = 1.15;

            camera.zoom = 1;
            camera.position.set(center.x, center.y, center.z + 100);

            if (size.x > aspect * size.y) {
                camera.left = (marginFactor * size.x) / -2;
                camera.right = (marginFactor * size.x) / 2;
                camera.top = (marginFactor * ((1 / aspect) * size.x)) / 2;
                camera.bottom = (marginFactor * ((1 / aspect) * size.x)) / -2;
            } else {
                camera.left = (marginFactor * (aspect * size.y)) / -2;
                camera.right = (marginFactor * (aspect * size.y)) / 2;
                camera.top = (marginFactor * size.y) / 2;
                camera.bottom = (marginFactor * size.y) / -2;
            }

            camera.updateProjectionMatrix();
            controls.target.copy(center);
            controls.update();
        },

        onResize(width: number, height: number): void {
            const aspect = width / height;
            camera.left = -aspect * 10;
            camera.right = aspect * 10;
            camera.top = 10;
            camera.bottom = -10;
            camera.updateProjectionMatrix();
        },

        dispose(): void {
            controls.dispose();
        },

        getState(): CameraState {
            return {
                position: [camera.position.x, camera.position.y, camera.position.z],
                target: [controls.target.x, controls.target.y, controls.target.z],
                zoom: camera.zoom,
                left: camera.left, right: camera.right,
                top: camera.top, bottom: camera.bottom,
            };
        },

        setState(state: CameraState): void {
            camera.position.set(...state.position);
            controls.target.set(...state.target);
            camera.zoom = state.zoom;
            camera.left = state.left; camera.right = state.right;
            camera.top = state.top; camera.bottom = state.bottom;
            camera.updateProjectionMatrix();
            controls.update();
        },
    };
}
