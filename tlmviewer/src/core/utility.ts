// Get an optional string key with a default value
// Also checks that the value is within the allowed options
// The default value is the first element of the options list
export function get_default(obj: any, key: string, options: string[]): string {
    const value = obj[key] ?? options[0];

    if (options.indexOf(value) == -1) {
        throw new Error(`${key} must be one of ${options}`);
    }

    return value;
}

// Get a required key that must have a value within a list of options
export function getOption<T extends string>(
    obj: any,
    key: string,
    options: readonly T[],
): T {
    if (!obj.hasOwnProperty(key)) {
        throw new Error(`missing required key '${key}'`);
    }
    const value = obj[key];
    if (options.indexOf(value) === -1) {
        throw new Error(`${key} must be one of ${options}`);
    }
    return value as T;
}

// Get a required key
export function getRequired<T>(obj: any, key: string): T {
    if (obj.hasOwnProperty(key)) {
        return obj[key] as T;
    } else {
        throw Error(`missing required key '${key}'`);
    }
}

// Like Array.map but for a 2D array
export function map2d<T>(
    array: number[][],
    f: (value: number, p: number, q: number) => T,
): T[][] {
    return array.map((row, p) => row.map((value, q) => f(value, p, q)));
}
