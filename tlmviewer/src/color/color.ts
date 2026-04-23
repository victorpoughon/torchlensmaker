export function floatRGBToHex(r: number, g: number, b: number): string {
    const toHex = (float: number): string => {
        const int: number = Math.round(float * 255);
        const hex: string = int.toString(16);
        return hex.length === 1 ? "0" + hex : hex;
    };

    return "#" + toHex(r) + toHex(g) + toHex(b);
}

export function colormap(x: number, lut: Array<Array<number>>): Array<number> {
    const N = lut.length;
    const index = Math.max(0, Math.min(Math.round(x * N), N - 1));

    const rgb = lut[index];
    return [rgb[0], rgb[1], rgb[2]];
}
