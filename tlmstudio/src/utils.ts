import type { Envelope } from 'tlmprotocol'
import { createHighlighterCore } from 'shiki/core'
import { createOnigurumaEngine } from 'shiki/engine/oniguruma'
import pythonLang from 'shiki/langs/python.mjs'
import githubDarkTheme from 'shiki/themes/github-dark.mjs'

export function getSceneName(envelope: Envelope & { type: 'scene' }) {
    const elements = envelope.payload.data

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

const highlighter = createHighlighterCore({
    langs: [pythonLang],
    themes: [githubDarkTheme],
    engine: createOnigurumaEngine(import('shiki/wasm')),
})

export async function highlightCode(content: string): Promise<string> {
    return (await highlighter).codeToHtml(content, { lang: 'python', theme: 'github-dark' })
}
