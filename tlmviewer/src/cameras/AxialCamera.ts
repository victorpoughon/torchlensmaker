import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type { CameraRig, CameraState } from "./CameraRig.ts";

export function createAxialCamera(
    axis: THREE.Vector3, // unit vector — orbit target stays on this axis through origin
    up: THREE.Vector3, // unit vector — screen-up direction
    domElement: HTMLElement,
): CameraRig {
    const camera = new THREE.OrthographicCamera(
        -10,
        10,
        10,
        -10,
        -10000,
        10000,
    );
    camera.up.copy(up);

    const controls = new OrbitControls(camera, domElement);
    controls.enablePan = false; // Replaced by custom axis pan below

    // Enforce target on the axis through origin after any orbit change
    controls.addEventListener("change", () => {
        controls.target.projectOnVector(axis);
    });

    // Custom right-click pan: move camera and target along the axis only.
    // We project the 2D screen-space drag vector onto the axis using the
    // camera's screen-right and screen-up directions, so panning always feels
    // natural regardless of orbit angle.
    let panStart = new THREE.Vector2();

    const onPointerDown = (e: PointerEvent) => {
        if (e.button === 2) panStart.set(e.clientX, e.clientY);
    };

    const onPointerMove = (e: PointerEvent) => {
        if (!(e.buttons & 2)) return; // right button not held

        const dx = e.clientX - panStart.x;
        const dy = e.clientY - panStart.y; // positive = downward in screen space
        panStart.set(e.clientX, e.clientY);

        const scale =
            (camera.right - camera.left) /
            (camera.zoom * domElement.clientWidth);

        // Camera basis vectors in world space.
        // cameraRight    = screen-right direction in world
        // cameraScreenUp = screen-up direction in world (NOT camera.up)
        const forward = new THREE.Vector3();
        camera.getWorldDirection(forward);
        const cameraRight = new THREE.Vector3()
            .crossVectors(forward, up)
            .normalize();
        const cameraScreenUp = new THREE.Vector3()
            .crossVectors(cameraRight, forward)
            .normalize();

        // Least-squares inverse mapping from screen drag (dx, dy) to axis shift.
        //
        // Moving the camera by Δ along `axis` shifts a fixed world point on screen by:
        //   Δscreen = (-cameraRight·axis · Δ, +cameraScreenUp·axis · Δ) / scale
        //
        // Solving for Δ given the observed drag (dx, dy):
        //   Δ = (-rDot·dx + uDot·dy) / (rDot² + uDot²) · scale
        //
        // The denominator is cos²(angle between view direction and axis). It amplifies
        // the shift at shallow angles so the drag always feels like grabbing the content.
        // It goes to zero when axis is pure depth (end-on view), in which case we skip.
        const rDot = cameraRight.dot(axis);
        const uDot = cameraScreenUp.dot(axis);
        const denom = rDot ** 2 + uDot ** 2;
        if (denom < 1e-6) return;
        const shift = ((-rDot * dx + uDot * dy) / denom) * scale;

        controls.target.addScaledVector(axis, shift);
        camera.position.addScaledVector(axis, shift);
        controls.update();
    };

    domElement.addEventListener("pointerdown", onPointerDown);
    domElement.addEventListener("pointermove", onPointerMove);

    return {
        camera,
        controls,

        fitToBox(bbox: THREE.Box3, aspect: number): void {
            const center = new THREE.Vector3();
            bbox.getCenter(center);
            const size = new THREE.Vector3();
            bbox.getSize(size);
            const marginFactor = 1.15;

            // Project bbox extents onto axis (horizontal) and up (vertical)
            const hsize =
                Math.abs(axis.x) * size.x +
                Math.abs(axis.y) * size.y +
                Math.abs(axis.z) * size.z;
            const vsize =
                Math.abs(up.x) * size.x +
                Math.abs(up.y) * size.y +
                Math.abs(up.z) * size.z;

            camera.zoom = 1;

            // Target: project bbox center onto axis through origin
            const t = axis.dot(center);
            controls.target.copy(axis).multiplyScalar(t);

            // Camera offset: perpendicular to both axis and up.
            // When axis = up, cross product is zero — fall back to any perpendicular.
            let offset = new THREE.Vector3().crossVectors(axis, up);
            if (offset.lengthSq() < 1e-10) {
                const ref =
                    Math.abs(axis.x) < 0.9
                        ? new THREE.Vector3(1, 0, 0)
                        : new THREE.Vector3(0, 1, 0);
                offset.crossVectors(axis, ref);
            }
            offset.normalize();
            camera.position.copy(controls.target).addScaledVector(offset, 100);

            if (hsize > aspect * vsize) {
                camera.left = (marginFactor * hsize) / -2;
                camera.right = (marginFactor * hsize) / 2;
                camera.top = (marginFactor * ((1 / aspect) * hsize)) / 2;
                camera.bottom = (marginFactor * ((1 / aspect) * hsize)) / -2;
            } else {
                camera.left = (marginFactor * (aspect * vsize)) / -2;
                camera.right = (marginFactor * (aspect * vsize)) / 2;
                camera.top = (marginFactor * vsize) / 2;
                camera.bottom = (marginFactor * vsize) / -2;
            }

            camera.updateProjectionMatrix();
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
            domElement.removeEventListener("pointerdown", onPointerDown);
            domElement.removeEventListener("pointermove", onPointerMove);
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

// Convenience factories — all 9 axis/up combinations

const X = new THREE.Vector3(1, 0, 0);
const Y = new THREE.Vector3(0, 1, 0);
const Z = new THREE.Vector3(0, 0, 1);

export const createAxialCameraXX = (d: HTMLElement) =>
    createAxialCamera(X, X, d);
export const createAxialCameraXY = (d: HTMLElement) =>
    createAxialCamera(X, Y, d);
export const createAxialCameraXZ = (d: HTMLElement) =>
    createAxialCamera(X, Z, d);
export const createAxialCameraYX = (d: HTMLElement) =>
    createAxialCamera(Y, X, d);
export const createAxialCameraYY = (d: HTMLElement) =>
    createAxialCamera(Y, Y, d);
export const createAxialCameraYZ = (d: HTMLElement) =>
    createAxialCamera(Y, Z, d);
export const createAxialCameraZX = (d: HTMLElement) =>
    createAxialCamera(Z, X, d);
export const createAxialCameraZY = (d: HTMLElement) =>
    createAxialCamera(Z, Y, d);
export const createAxialCameraZZ = (d: HTMLElement) =>
    createAxialCamera(Z, Z, d);

export const axialFactories: Record<string, (d: HTMLElement) => CameraRig> = {
    "axial-xx": createAxialCameraXX,
    "axial-xy": createAxialCameraXY,
    "axial-xz": createAxialCameraXZ,
    "axial-yx": createAxialCameraYX,
    "axial-yy": createAxialCameraYY,
    "axial-yz": createAxialCameraYZ,
    "axial-zx": createAxialCameraZX,
    "axial-zy": createAxialCameraZY,
    "axial-zz": createAxialCameraZZ,
};
