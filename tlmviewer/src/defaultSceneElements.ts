export function defaultSceneElementsData(dim: number): any[] {
    if (dim === 2) {
        return [
            { type: "scene-axis", axis: "x", length: 500, color: "#e3e3e3" },
            { type: "scene-axis", axis: "y", length: 500, color: "#e3e3e3" },
            { type: "ambient-light", color: "#ffffff", intensity: 0.5 },
            {
                type: "directional-light",
                color: "#ffffff",
                intensity: 0.5,
                position: [100, 100, 100],
            },
        ];
    } else {
        return [
            { type: "scene-axis", axis: "x", length: 500, color: "#e3e3e3" },
            { type: "scene-axis", axis: "y", length: 10, color: "#C80000" },
            { type: "scene-axis", axis: "z", length: 10, color: "#00C800" },
            { type: "ambient-light", color: "#ffffff", intensity: 0.5 },
            {
                type: "directional-light",
                color: "#ffffff",
                intensity: 0.5,
                position: [100, 100, 100],
            },
        ];
    }
}
