import * as THREE from "three";
import { GUI, Controller } from "lil-gui";

import { ColorOption } from "./core/events.ts";
import { TLMScene } from "./scene.ts";
import { TLMViewerApp } from "./app.ts";

type RGBColor = {
    r: number;
    g: number;
    b: number;
};

export class TLMGui {
    // @ts-ignore
    private app: TLMViewerApp;
    private scene: TLMScene;
    private controller: any;
    private gui: GUI;

    private colorOptions: Record<string, ColorOption>;

    // Controllers
    private controllers: {
        colors: {
            validRays: Controller;
            blockedRays: Controller;
            outputRays: Controller;
            opacity: Controller;
            thickness: Controller;
            background: Controller;
            surfaces: Controller;
        };
        visible: {
            validRays: Controller;
            blockedRays: Controller;
            outputRays: Controller;
            surfaces: Controller;
            axisX: Controller;
            axisY: Controller;
            axisZ: Controller;
            kinematicJoints: Controller;
            bcyl: Controller;
        };
    };

    // GUI folders
    private folders: {
        colors: GUI;
        show: GUI;
    };

    constructor(app: TLMViewerApp, container: HTMLElement, scene: TLMScene) {
        this.scene = scene;
        this.app = app;
        this.gui = new GUI({ container: container, autoPlace: false });

        // Build this.colorOptions from the data
        this.colorOptions = {
            default: { colorDim: null, trueColor: false },
        };

        for (const varName of this.scene.variables) {
            if (varName === "wavelength") {
                this.colorOptions["wavelength"] = {
                    colorDim: "wavelength",
                    trueColor: false,
                };
                this.colorOptions["wavelength (true color)"] = {
                    colorDim: "wavelength",
                    trueColor: true,
                };
            } else {
                this.colorOptions[varName] = {
                    colorDim: varName,
                    trueColor: false,
                };
            }
        }

        this.controller = {
            validColor: this.colorOptions.default,
            blockedColor: this.colorOptions.default,
            outputColor: this.colorOptions.default,
            raysOpacity: 1.0,
            raysThickness: 1.0,
            camera: app.cameraType,
            resetView() {
                app.resetView();
            },
            backgroundColor: { r: 0, g: 0, b: 0 },
            surfacesColor: { r: 0, g: 1, b: 1 },

            showAxisX: false,
            showAxisY: false,
            showAxisZ: false,

            showValidRays: true,
            showBlockedRays: false,
            showOutputRays: true,
            showSurfaces: true,
            showKinematicJoints: false,
            showBcyl: false,
        };

        // If 'field' variable is available, default to it for valid and output rays
        if (this.scene.variables.includes("field")) {
            this.controller.validColor = this.colorOptions["field"];
            this.controller.outputColor = this.colorOptions["field"];
        }

        this.gui.add(this.controller, "resetView").name("Reset Camera");

        this.gui
            .add(this.controller, "camera", {
                "2D": "2D",
                Orthographic: "orthographic",
                Perspective: "perspective",
                "Axial X (↑ X)": "axial-xx",
                "Axial X (↑ Y)": "axial-xy",
                "Axial X (↑ Z)": "axial-xz",
                "Axial Y (↑ X)": "axial-yx",
                "Axial Y (↑ Y)": "axial-yy",
                "Axial Y (↑ Z)": "axial-yz",
                "Axial Z (↑ X)": "axial-zx",
                "Axial Z (↑ Y)": "axial-zy",
                "Axial Z (↑ Z)": "axial-zz",
            })
            .name("Camera")
            .onChange((value: string) => {
                app.setCamera(value);
            });

        const folderColors = this.gui.addFolder("Colors");

        const controllerColorsValidRays = folderColors
            .add(this.controller, "validColor", this.colorOptions)
            .name("Valid rays");

        const controllerColorsBlockedRays = folderColors
            .add(this.controller, "blockedColor", this.colorOptions)
            .name("Blocked rays");

        const controllerColorsOutputRays = folderColors
            .add(this.controller, "outputColor", this.colorOptions)
            .name("Output rays");

        const controllerColorsOpacity = folderColors
            .add(this.controller, "raysOpacity", 0, 1)
            .name("Opacity")
            .onFinishChange((value: number) => {
                this.scene.dispatch({ type: "setRaysOpacity", value: value });
            });

        const controllerColorsThickness = folderColors
            .add(this.controller, "raysThickness", 0.1, 10)
            .name("Thickness")
            .onFinishChange((value: number) => {
                this.scene.dispatch({ type: "setRaysThickness", value: value });
            });

        controllerColorsValidRays.onChange((value: ColorOption) => {
            this.scene.dispatch({ type: "setValidRaysColor", value });
            this.scene.dispatch({
                type: "setRaysOpacity",
                value: controllerColorsOpacity.getValue(),
            });
            this.scene.dispatch({
                type: "setRaysThickness",
                value: controllerColorsThickness.getValue(),
            });
        });

        controllerColorsBlockedRays.onChange((value: ColorOption) => {
            this.scene.dispatch({ type: "setBlockedRaysColor", value });
            this.scene.dispatch({
                type: "setRaysOpacity",
                value: controllerColorsOpacity.getValue(),
            });
            this.scene.dispatch({
                type: "setRaysThickness",
                value: controllerColorsThickness.getValue(),
            });
        });

        controllerColorsOutputRays.onChange((value: ColorOption) => {
            this.scene.dispatch({ type: "setOutputRaysColor", value });
            this.scene.dispatch({
                type: "setRaysOpacity",
                value: controllerColorsOpacity.getValue(),
            });
            this.scene.dispatch({
                type: "setRaysThickness",
                value: controllerColorsThickness.getValue(),
            });
        });

        const controllerColorsBackground = folderColors
            .addColor(this.controller, "backgroundColor")
            .name("Background")
            .onChange((value: RGBColor) => {
                this.scene.scene.background = new THREE.Color().setRGB(
                    value.r,
                    value.g,
                    value.b,
                    THREE.SRGBColorSpace,
                );
            });

        const controllerColorsSurfaces = folderColors
            .addColor(this.controller, "surfacesColor")
            .name("Surfaces")
            .onChange((value: RGBColor) => {
                const color = new THREE.Color(value.r, value.g, value.b);
                this.scene.dispatch({ type: "setSurfacesColor", value: color });
            });

        const folderShow = this.gui.addFolder("Visible");
        const controllerVisibleValidRays = folderShow
            .add(this.controller, "showValidRays")
            .name("Valid rays")
            .onChange((value: boolean) => {
                this.scene.dispatch({
                    type: "setCategoryVisibility",
                    category: "rays-valid",
                    visible: value,
                });
            });
        const controllerVisibleBlockedRays = folderShow
            .add(this.controller, "showBlockedRays")
            .name("Blocked rays")
            .onChange((value: boolean) => {
                this.scene.dispatch({
                    type: "setCategoryVisibility",
                    category: "rays-blocked",
                    visible: value,
                });
            });
        const controllerVisibleOutputRays = folderShow
            .add(this.controller, "showOutputRays")
            .name("Output rays")
            .onChange((value: boolean) => {
                this.scene.dispatch({
                    type: "setCategoryVisibility",
                    category: "rays-output",
                    visible: value,
                });
            });
        const controllerVisibleSurfaces = folderShow
            .add(this.controller, "showSurfaces")
            .name("Surfaces")
            .onChange((value: boolean) => {
                this.scene.dispatch({
                    type: "setCategoryVisibility",
                    category: "surface",
                    visible: value,
                });
            });
        const controllerVisibleAxisX = folderShow
            .add(this.controller, "showAxisX")
            .name("Axis X")
            .onChange((value: boolean) => {
                this.scene.dispatch({
                    type: "setCategoryVisibility",
                    category: "axis-x",
                    visible: value,
                });
            });
        const controllerVisibleAxisY = folderShow
            .add(this.controller, "showAxisY")
            .name("Axis Y")
            .onChange((value: boolean) => {
                this.scene.dispatch({
                    type: "setCategoryVisibility",
                    category: "axis-y",
                    visible: value,
                });
            });
        const controllerVisibleAxisZ = folderShow
            .add(this.controller, "showAxisZ")
            .name("Axis Z")
            .onChange((value: boolean) => {
                this.scene.dispatch({
                    type: "setCategoryVisibility",
                    category: "axis-z",
                    visible: value,
                });
            });
        const controllerVisibleKinematicJoints = folderShow
            .add(this.controller, "showKinematicJoints")
            .name("Kinematic joints")
            .onChange((value: boolean) => {
                this.scene.dispatch({
                    type: "setCategoryVisibility",
                    category: "kinematic-joint",
                    visible: value,
                });
            });
        const controllerVisibleBcyl = folderShow
            .add(this.controller, "showBcyl")
            .name("Bounding Cylinders")
            .onChange((value: boolean) => {
                this.scene.dispatch({
                    type: "setCategoryVisibility",
                    category: "bcyl",
                    visible: value,
                });
            });
        // Initialize this.controllers
        this.controllers = {
            colors: {
                validRays: controllerColorsValidRays,
                blockedRays: controllerColorsBlockedRays,
                outputRays: controllerColorsOutputRays,
                opacity: controllerColorsOpacity,
                thickness: controllerColorsThickness,
                background: controllerColorsBackground,
                surfaces: controllerColorsSurfaces,
            },
            visible: {
                validRays: controllerVisibleValidRays,
                blockedRays: controllerVisibleBlockedRays,
                outputRays: controllerVisibleOutputRays,
                surfaces: controllerVisibleSurfaces,
                axisX: controllerVisibleAxisX,
                axisY: controllerVisibleAxisY,
                axisZ: controllerVisibleAxisZ,
                kinematicJoints: controllerVisibleKinematicJoints,
                bcyl: controllerVisibleBcyl,
            },
        };

        // Initialize this.folder
        this.folders = {
            colors: folderColors,
            show: folderShow,
        };

        this.setDefaultGUIState();
    }

    public setDefaultGUIState() {
        // Set default GUI state
        this.scene.dispatch({
            type: "setValidRaysColor",
            value: this.controller.validColor,
        });
        this.scene.dispatch({
            type: "setBlockedRaysColor",
            value: this.controller.blockedColor,
        });
        this.scene.dispatch({
            type: "setOutputRaysColor",
            value: this.controller.outputColor,
        });

        this.scene.dispatch({
            type: "setCategoryVisibility",
            category: "rays-valid",
            visible: true,
        });
        this.scene.dispatch({
            type: "setCategoryVisibility",
            category: "rays-blocked",
            visible: false,
        });
        this.scene.dispatch({
            type: "setCategoryVisibility",
            category: "rays-output",
            visible: true,
        });
        this.scene.dispatch({
            type: "setCategoryVisibility",
            category: "axis-x",
            visible: false,
        });
        this.scene.dispatch({
            type: "setCategoryVisibility",
            category: "axis-y",
            visible: false,
        });
        this.scene.dispatch({
            type: "setCategoryVisibility",
            category: "axis-z",
            visible: false,
        });
        this.scene.dispatch({
            type: "setCategoryVisibility",
            category: "kinematic-joint",
            visible: false,
        });
        this.scene.dispatch({
            type: "setCategoryVisibility",
            category: "bcyl",
            visible: false,
        });

        this.gui.open(false);
        this.folders.colors.open(true);
        this.folders.show.open(true);
    }

    // Set controls state from a JSON object
    public setControlsFromJson(controls: any) {
        const set = function (key: string, setter: any) {
            if (controls.hasOwnProperty(key)) {
                setter(controls[key]);
            }
        };
        const self = this;

        set("color_rays", (v: string) => {
            self.controllers.colors.validRays.load(self.colorOptions[v]);
            self.controllers.colors.outputRays.load(self.colorOptions[v]);
        });
        set("valid_rays", (v: string) => {
            self.controllers.colors.validRays.load(self.colorOptions[v]);
        });
        set("blocked_rays", (v: string) => {
            self.controllers.colors.blockedRays.load(self.colorOptions[v]);
        });
        set("output_rays", (v: string) => {
            self.controllers.colors.outputRays.load(self.colorOptions[v]);
        });
        set("opacity", (v: number) => {
            self.controllers.colors.opacity.load(v);
        });
        set("thickness", (v: number) => {
            self.controllers.colors.thickness.load(v);
        });
        set("show_valid_rays", (v: boolean) => {
            self.controllers.visible.validRays.load(v);
        });
        set("show_blocked_rays", (v: boolean) => {
            self.controllers.visible.blockedRays.load(v);
        });
        set("show_output_rays", (v: boolean) => {
            self.controllers.visible.outputRays.load(v);
        });
        set("show_surfaces", (v: boolean) => {
            self.controllers.visible.surfaces.load(v);
        });
        set("show_axis_x", (v: boolean) => {
            self.controllers.visible.axisX.load(v);
        });
        set("show_axis_y", (v: boolean) => {
            self.controllers.visible.axisY.load(v);
        });
        set("show_axis_z", (v: boolean) => {
            self.controllers.visible.axisZ.load(v);
        });
        set("show_kinematic_joints", (v: boolean) => {
            self.controllers.visible.kinematicJoints.load(v);
        });
        set("show_bounding_cylinders", (v: boolean) => {
            self.controllers.visible.bcyl.load(v);
        });

        // Finish with visibility of the gui itself
        set("show_controls", (v: boolean) => {
            if (v) {
                self.gui.show();
            } else {
                self.gui.hide();
            }
        });
    }
}
