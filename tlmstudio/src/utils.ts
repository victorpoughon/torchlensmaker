import type { Envelope, SceneData } from 'tlmprotocol'

export function getSceneName(envelope: Envelope & { type: 'scene' }) {
    const elements = (envelope.payload as SceneData).data

    const firstTitle = elements.find((el) => el.type === 'scene-title')

    if (firstTitle === undefined) {
        return `Untitled scene (${envelope.topic})`
    } else {
        return firstTitle.title
    }
}
