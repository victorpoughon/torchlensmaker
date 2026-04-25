import type { SceneData } from './scene.ts'
import type { SourcePayload } from './source.ts'

export const PROTOCOL_VERSION = 2;

export type Envelope =
    | { v: number; type: 'scene';  topic: string; payload: SceneData }
    | { v: number; type: 'source'; topic: string; payload: SourcePayload }
    | { v: number; type: 'log';    topic: string; payload: string }
    | { v: number; type: 'plot';   topic: string; payload: unknown }
    | { v: number; type: 'image';  topic: string; payload: unknown }
