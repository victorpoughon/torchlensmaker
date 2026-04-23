import { ElementDescriptor } from "../core/types.ts";

import { ambientLightDescriptor } from "../elements_basic/AmbientLight.ts";
import { directionalLightDescriptor } from "../elements_basic/DirectionalLight.ts";
import { sceneAxisDescriptor } from "../elements_basic/SceneAxis.ts";
import { arrowsDescriptor } from "../elements_basic/ArrowsElement.ts";
import { box3DDescriptor } from "../elements_basic/Box3D.ts";
import { pointsDescriptor } from "../elements_basic/PointsElement.ts";
import { cylinderDescriptor } from "../elements_basic/Cylinder.ts";
import { raysDescriptor } from "../elements_basic/RaysElement.ts";
import { surfaceDiskDescriptor } from "../elements_surfaces/SurfaceDisk.ts";
import { surfaceLatheDescriptor } from "../elements_surfaces/SurfaceLathe.ts";
import { surfaceSphereRDescriptor } from "../elements_surfaces/SurfaceSphereR.ts";
import { surfaceSagDescriptor } from "../elements_surfaces/SurfaceSag.ts";
import { surfaceBSplineDescriptor } from "../elements_surfaces/SurfaceBSpline.ts";
import { sceneTitleDescriptor } from "../elements_basic/SceneTitle.ts";

export type { SceneElementData } from "tlmprotocol";

// List of all scene elements descriptors
export const allDescriptors = [
    ambientLightDescriptor,
    arrowsDescriptor,
    box3DDescriptor,
    cylinderDescriptor,
    directionalLightDescriptor,
    pointsDescriptor,
    raysDescriptor,
    sceneAxisDescriptor,
    sceneTitleDescriptor,
    surfaceLatheDescriptor,
    surfaceDiskDescriptor,
    surfaceSagDescriptor,
    surfaceSphereRDescriptor,
    surfaceBSplineDescriptor,
] as const;

const descriptorMap = new Map<string, ElementDescriptor<any>>(
    allDescriptors.map((d) => [d.type, d]),
);

export function getDescriptor(type: string): ElementDescriptor<any> {
    const desc = descriptorMap.get(type);
    if (!desc) throw new Error(`Unknown element type: "${type}"`);
    return desc;
}

export function getMaybeDescriptor(
    type: string,
): ElementDescriptor<any> | undefined {
    const desc = descriptorMap.get(type);
    return desc;
}
