import * as THREE from "three";

import { TLMScene } from "./scene.ts";
import { TLMGui } from "./gui.ts";
import type { CameraRig, CameraState } from "./cameras/CameraRig.ts";
import { createCamera2D } from "./cameras/Camera2D.ts";
import { createOrthographicCamera } from "./cameras/OrthographicCamera.ts";
import { createPerspectiveCamera } from "./cameras/PerspectiveCamera.ts";
import { axialFactories } from "./cameras/AxialCamera.ts";

import "./viewer.css";

export class TLMViewerApp {
    public scene: TLMScene;

    public renderer: THREE.WebGLRenderer;
    public rig: CameraRig;
    public cameraType: string;
    public viewport: HTMLElement;
    public gui: TLMGui;

    constructor(container: HTMLElement, scene: TLMScene, camera: string, cameraState?: CameraState) {
        const viewport =
            container.getElementsByClassName("tlmviewer-viewport")[0];

        if (!(viewport instanceof HTMLElement))
            throw new Error(`Expected viewport to be an HTMLElement`);

        this.viewport = viewport;
        this.scene = scene;

        // Set up the renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        const rect = container.getBoundingClientRect();
        this.renderer.setSize(rect.width, rect.height);
        this.renderer.localClippingEnabled = true;
        this.viewport.appendChild(this.renderer.domElement);

        // Setup the camera rig
        this.cameraType = camera;
        this.rig = this.createRig(camera);

        // LIL GUI
        this.gui = new TLMGui(this, container, this.scene);

        if (cameraState) {
            this.rig.setState(cameraState);
        } else {
            this.resetView();
        }
    }

    private createRig(cameraType: string): CameraRig {
        const domElement = this.renderer.domElement;
        if (cameraType === "2D") {
            return createCamera2D(domElement);
        } else if (cameraType === "orthographic") {
            return createOrthographicCamera(domElement);
        } else if (cameraType === "perspective") {
            return createPerspectiveCamera(domElement);
        } else if (axialFactories[cameraType]) {
            return axialFactories[cameraType](domElement);
        } else {
            throw new Error(`Unknown camera type '${cameraType}'`);
        }
    }

    public onWindowResize(): void {
        const rect = this.viewport.getBoundingClientRect();
        this.rig.onResize(rect.width, rect.height);
        this.renderer.setSize(rect.width, rect.height);
    }

    public resetView(): void {
        const bbox = this.scene.getBB();
        const rect = this.viewport.getBoundingClientRect();
        const aspect = rect.width / rect.height;
        this.rig.fitToBox(bbox, aspect);
    }

    public setCamera(type: string): void {
        this.rig.dispose();
        this.cameraType = type;
        this.rig = this.createRig(type);
        const rect = this.viewport.getBoundingClientRect();
        this.renderer.setSize(rect.width, rect.height);
        this.rig.onResize(rect.width, rect.height);
        this.resetView();
    }

    public registerEventHandlers(container: HTMLElement): void {
        const resetViewButton = container.querySelector("button.reset-view");
        resetViewButton?.addEventListener("click", () => {
            this.resetView();
        });
    }

    public dispose(): void {
        this.rig.dispose();
        this.renderer.dispose();
    }

    public showFps(visible: boolean): void {
        const el = this.viewport.parentElement?.querySelector(".tlmviewer-fps");
        el?.classList.toggle("visible", visible);
    }

    public animate(): () => void {
        let animId: number;
        let frameCount = 0;
        let lastFpsTime = performance.now();

        const fpsEl = this.viewport.parentElement?.querySelector(".tlmviewer-fps");

        const loop = () => {
            this.rig.controls.update();
            this.renderer.render(this.scene.scene, this.rig.camera);

            frameCount++;
            const now = performance.now();
            const elapsed = now - lastFpsTime;
            if (elapsed >= 500) {
                const fps = Math.round(frameCount * 1000 / elapsed);
                if (fpsEl) fpsEl.textContent = `${fps} fps`;
                frameCount = 0;
                lastFpsTime = now;
            }

            animId = requestAnimationFrame(loop);
        };
        animId = requestAnimationFrame(loop);
        return () => cancelAnimationFrame(animId);
    }
}
