import tlmviewer from "tlmviewer";

const { allDescriptors } = tlmviewer.testing;

function buildScene(
    sceneName: string,
    mode: "2D" | "3D",
    elements: any[],
): { sceneName: string; data: object } {
    const camera = mode === "2D" ? "2D" : "orthographic";

    // Add title
    const title = { type: "scene-title", title: sceneName };

    return {
        sceneName: sceneName,
        data: {
            mode,
            camera,
            data: [...elements, title],
            controls: {
                show_x_axis: "true",
                show_y_axis: "true",
                show_z_axis: "true",
                show_bounding_cylinders: "true",
            },
        },
    };
}

// Generate test scenes from the new registry
const registryScenes: Array<{ sceneName: string; data: object }> = [];
for (const descriptor of allDescriptors) {
    if (descriptor.testData2D.length > 0) {
        registryScenes.push(
            buildScene(`${descriptor.type}/2D`, "2D", descriptor.testData2D),
        );
    }
    if (descriptor.testData3D.length > 0) {
        registryScenes.push(
            buildScene(`${descriptor.type}/3D`, "3D", descriptor.testData3D),
        );
    }
}

export const builtinScenes: Array<{ sceneName: string; data: object }> = [
    ...registryScenes,
];
