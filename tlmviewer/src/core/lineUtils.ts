import * as THREE from "three";

import { Line2 } from "three/addons/lines/Line2.js";
import { LineGeometry } from "three/addons/lines/LineGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";

export function makeLine(start: number[], end: number[], color: string) {
    console.assert(start.length == 3);
    console.assert(end.length == 3);

    const material = new THREE.LineBasicMaterial({ color: color });

    const points = [];
    points.push(new THREE.Vector3().fromArray(start));
    points.push(new THREE.Vector3().fromArray(end));

    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    return new THREE.Line(geometry, material);
}

export function makeLine2(start: number[], end: number[], color: string) {
    console.assert(start.length == 3);
    console.assert(end.length == 3);

    const material = new LineMaterial({
        color: color,
        linewidth: 1.1,
        worldUnits: false,
        side: THREE.DoubleSide,
        transparent: true,
    });

    const points = [];
    points.push(new THREE.Vector3().fromArray(start));
    points.push(new THREE.Vector3().fromArray(end));

    const geometry = new LineGeometry().setFromPoints(points);
    const ret = new Line2(geometry, material);

    return ret;
}
