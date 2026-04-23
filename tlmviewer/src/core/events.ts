import * as THREE from "three";

export type ColorOption = {
    colorDim: string | null;
    trueColor: boolean;
};

export type SceneEventMap = {
    setValidRaysColor: { value: ColorOption };
    setBlockedRaysColor: { value: ColorOption };
    setOutputRaysColor: { value: ColorOption };
    setRaysOpacity: { value: number };
    setRaysThickness: { value: number };
    setSurfacesColor: { value: THREE.Color };
    setCategoryVisibility: { category: string; visible: boolean };
    setBackground: { value: THREE.Color }; // TODO
    resetView: {}; // TODO
};

export type SceneEventType = keyof SceneEventMap;

export type SceneEvent<K extends SceneEventType = SceneEventType> = {
    type: K;
} & SceneEventMap[K];
