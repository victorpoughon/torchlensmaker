import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export type CameraState = {
    position: [number, number, number];
    target: [number, number, number];
    zoom: number;
    left: number;
    right: number;
    top: number;
    bottom: number;
};

export interface CameraRig {
    camera: THREE.PerspectiveCamera | THREE.OrthographicCamera;
    controls: OrbitControls;

    fitToBox(bbox: THREE.Box3, viewportAspect: number): void;
    onResize(width: number, height: number): void;
    dispose(): void;
    getState(): CameraState;
    setState(state: CameraState): void;
}
