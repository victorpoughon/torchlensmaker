export const PROTOCOL_VERSION = 2;

export type MessageType = "scene" | "plot" | "log" | "image" | "source";

export type Envelope<T = unknown> = {
    v: number;
    type: MessageType;
    topic: string;
    payload: T;
};
