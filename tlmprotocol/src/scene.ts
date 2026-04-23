export type BaseElementData = {
    type: string;
};

export type SurfaceBaseData = BaseElementData & {
    matrix: number[][];
    clipPlanes: [number, number, number, number][];
};

export type AmbientLightData = BaseElementData & {
    type: "ambient-light";
    color: string;
    intensity: number;
};

export type ArrowsData = BaseElementData & {
    type: "arrows";
    arrows: number[][];
};

export type Box3DData = BaseElementData & {
    type: "box3D";
    size: [number, number, number];
    matrix: number[][];
};

export type CylinderData = BaseElementData & {
    type: "cylinder";
    matrix: number[][];
    xmin: number;
    xmax: number;
    radius: number;
};

export type DirectionalLightData = BaseElementData & {
    type: "directional-light";
    color: string;
    intensity: number;
    position: [number, number, number];
};

export type PointsData = BaseElementData & {
    type: "points";
    vertices: number[][];
    color: string;
    radius: number;
    category: string;
};

export type RaysData = BaseElementData & {
    type: "rays";
    points: number[][];
    color: string;
    variables: Record<string, number[]>;
    domain: Record<string, [number, number]>;
    dim: 2 | 3;
    category: string;
};

export type SceneAxisData = BaseElementData & {
    type: "scene-axis";
    axis: "x" | "y" | "z";
    length: number;
    color: string;
};

export type SceneTitleData = BaseElementData & {
    type: "scene-title";
    title: string;
};

export type SurfaceDiskData = SurfaceBaseData & {
    type: "surface-disk";
    radius: number;
};

export type SurfaceLatheData = SurfaceBaseData & {
    type: "surface-lathe";
    samples: number[][];
};

export type SurfaceSagData = SurfaceBaseData & {
    type: "surface-sag";
    diameter: number;
    sagFunctionData: unknown;
};

export type SurfaceSphereRData = SurfaceBaseData & {
    type: "surface-sphere-r";
    R: number;
    diameter: number;
};

export type SurfaceBSplineData = SurfaceBaseData & {
    type: "surface-bspline";
    points: number[][][];
    weights: number[][];
    degree: [number, number];
    knotType: "clamped" | "unclamped";
    samples: [number, number];
};

export type SceneElementData =
    | AmbientLightData
    | ArrowsData
    | Box3DData
    | CylinderData
    | DirectionalLightData
    | PointsData
    | RaysData
    | SceneAxisData
    | SceneTitleData
    | SurfaceLatheData
    | SurfaceDiskData
    | SurfaceSagData
    | SurfaceSphereRData
    | SurfaceBSplineData;

export type SceneData = {
    data: SceneElementData[];
    mode?: "3D" | "2D";
    camera?: "orthographic" | "perspective" | "2D";
    controls?: Record<string, unknown>;
};
