import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type { CameraRig, CameraState } from "./CameraRig.ts";

export function createPerspectiveCamera(domElement: HTMLElement): CameraRig {
    const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
    camera.position.set(10, 10, 10);
    camera.lookAt(0, 0, 0);

    const controls = new OrbitControls(camera, domElement);

    return {
        camera,
        controls,

        fitToBox(bbox: THREE.Box3, aspect: number): void {
            const center = new THREE.Vector3();
            bbox.getCenter(center);
            const size = new THREE.Vector3();
            bbox.getSize(size);
            const marginFactor = 1.15;

            // Compute the camera distance so the bounding box fits within the frustum.
            // We check both the vertical and horizontal extents and take the larger.
            const vFov = camera.fov * (Math.PI / 180);
            const hFov = 2 * Math.atan(Math.tan(vFov / 2) * aspect);

            const distForHeight = size.y / 2 / Math.tan(vFov / 2);
            const distForWidth = size.x / 2 / Math.tan(hFov / 2);
            // Also pull back enough to clear the depth of the scene
            const distance =
                Math.max(distForHeight, distForWidth, size.z / 2) * marginFactor;

            camera.position.set(center.x, center.y, center.z + distance);
            controls.target.copy(center);
            controls.update();
        },

        onResize(width: number, height: number): void {
            camera.aspect = width / height;
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
                left: 0, right: 0, top: 0, bottom: 0,
            };
        },

        setState(state: CameraState): void {
            camera.position.set(...state.position);
            controls.target.set(...state.target);
            controls.update();
        },
    };
}
