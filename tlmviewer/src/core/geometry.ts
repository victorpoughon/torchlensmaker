import * as THREE from "three";

export function arrayToMatrix4(array: Array<Array<number>>): THREE.Matrix4 {
    if (array.length !== 4 || array.some((row) => row.length !== 4)) {
        throw new Error("Input must be a 4x4 array");
    }

    const matrix = new THREE.Matrix4();

    // Transpose the array (convert from row-major to column-major)
    const transposedArray = array[0].map((_, colIndex) =>
        array.map((row) => row[colIndex]),
    );

    // Flatten the transposed array and create the Matrix4
    matrix.fromArray(transposedArray.flat());

    return matrix;
}

// Convert a 3x3 homogeneous matrix to a 4x4 homogeneous matrix,
// by inserting identity transform to the new axis
export function homogeneousMatrix3to4(matrix: number[][]): number[][] {
    // Check if the input matrix is 3x3
    if (matrix.length !== 3 || matrix.some((row) => row.length !== 3)) {
        throw new Error("Input matrix must be 3x3");
    }

    return [
        [matrix[0][0], matrix[0][1], 0, matrix[0][2]],
        [matrix[1][0], matrix[1][1], 0, matrix[1][2]],
        [0, 0, 1, 0],
        [matrix[2][0], matrix[2][1], 0, matrix[2][2]],
    ];
}

export function samples2DToPoints(samples: Array<Array<number>>) {
    const points: number[] = [];
    for (const p of samples) {
        points.push(p[0], p[1], 1.0);
    }
    return points;
}

// Derive a THREE.Matrix4 from a 3x3 homogeneous (2D) surface matrix
export function getTransform2D(matrix: number[][]): THREE.Matrix4 {
    return arrayToMatrix4(homogeneousMatrix3to4(matrix));
}

// Derive a THREE.Matrix4 from a 4x4 (3D) surface matrix
export function getTransform3D(matrix: number[][]): THREE.Matrix4 {
    return arrayToMatrix4(matrix);
}
