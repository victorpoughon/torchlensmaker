import type { Envelope, SceneData } from 'tlmprotocol'
import { codeToHtml } from 'shiki'

export function getSceneName(envelope: Envelope & { type: 'scene' }) {
    const elements = (envelope.payload as SceneData).data

    const firstTitle = elements.find((el) => el.type === 'scene-title')

    if (firstTitle === undefined) {
        return `Untitled scene (${envelope.topic})`
    } else {
        return firstTitle.title
    }
}

export function formatTime(date: Date): string {
    return date.toLocaleTimeString()
}

export async function highlightCode(content: string, language: string): Promise<string> {
    try {
        return await codeToHtml(content, { lang: language, theme: 'github-dark' })
    } catch {
        return await codeToHtml(content, { lang: 'text', theme: 'github-dark' })
    }
}
