import { TLMScene } from "./scene.ts";
import { TLMViewerApp } from "./app.ts";
import { get_default } from "./core/utility.ts";
import type { CameraState } from "./cameras/CameraRig.ts";
import viewerTemplate from "./viewer.html?raw";

export type RenderHandle = {
    getCameraState(): CameraState;
    dispose(): void;
};

export function renderScene(
    container: HTMLElement,
    data: unknown,
    cameraState?: CameraState,
): RenderHandle {
    try {
        const d = data as any;
        container.innerHTML = viewerTemplate;

        const mode = get_default(d, "mode", ["3D", "2D"]);
        const camera = get_default(d, "camera", [
            "orthographic",
            "perspective",
            "2D",
        ]);

        const scene = new TLMScene(d, mode === "3D" ? 3 : 2, container);
        const app = new TLMViewerApp(container, scene, camera, cameraState);

        const controls = d["controls"] ?? {};
        app.gui.setControlsFromJson(controls);

        const onResize = () => app.onWindowResize();
        window.addEventListener("resize", onResize);
        app.registerEventHandlers(container);
        const cancelAnimation = app.animate();

        return {
            getCameraState: () => app.rig.getState(),
            dispose: () => {
                cancelAnimation();
                window.removeEventListener("resize", onResize);
                app.dispose();
            },
        };
    } catch (error) {
        container.innerHTML =
            "<span style='color: red'>tlmviewer error: " + error + "</span>";
        throw error;
    }
}
