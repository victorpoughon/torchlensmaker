import { PROTOCOL_VERSION } from "tlmprotocol";
import type { MessageType } from "tlmprotocol";
import { renderScene } from "./render.ts";
import type { RenderHandle } from "./render.ts";

export interface ConnectOptions {
    topic?: string;
    type?: MessageType;
}

function showStatus(container: HTMLElement, message: string): void {
    container.innerHTML = `<div style="width:100%;height:100%;display:flex;align-items:center;justify-content:center;background:#111;color:#888;font-family:monospace;font-size:14px;">${message}</div>`;
}

export function connect(
    container: HTMLElement,
    wsUrl: string,
    opts: ConnectOptions = {},
): () => void {
    const { topic, type: filterType } = opts;
    let ws: WebSocket | null = null;
    let reconnectDelay = 1000;
    let stopped = false;
    let hasScene = false;
    let handle: RenderHandle | null = null;

    function connectWs(): void {
        if (!hasScene) showStatus(container, "Connecting…");

        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            reconnectDelay = 1000;
            if (!hasScene) showStatus(container, "Waiting for scene…");
        };

        ws.onmessage = (event) => {
            let envelope: any;
            try {
                envelope = JSON.parse(event.data as string);
            } catch (e) {
                console.error("tlmviewer: failed to parse message", e);
                return;
            }

            if (envelope.v !== PROTOCOL_VERSION) {
                console.warn(
                    `tlmviewer: protocol version mismatch (expected ${PROTOCOL_VERSION}, got ${envelope.v})`,
                );
            }

            if (topic !== undefined && envelope.topic !== topic) return;
            if (filterType !== undefined && envelope.type !== filterType) return;

            console.log(`tlmviewer: received topic=${envelope.topic} type=${envelope.type} v=${envelope.v}`);

            if (envelope.type === "scene") {
                hasScene = true;
                const savedState = handle?.getCameraState();
                handle?.dispose();
                try {
                    handle = renderScene(container, envelope.payload, savedState);
                } catch (e) {
                    console.error("tlmviewer: renderScene failed", e);
                }
            }
        };

        ws.onerror = (e) => {
            console.error("tlmviewer: WebSocket error", e);
            ws?.close();
        };

        ws.onclose = () => {
            if (stopped) return;
            if (!hasScene)
                showStatus(container, `Reconnecting in ${reconnectDelay / 1000}s…`);
            setTimeout(connectWs, reconnectDelay);
            reconnectDelay = Math.min(reconnectDelay * 2, 10000);
        };
    }

    connectWs();

    return () => {
        stopped = true;
        ws?.close();
        handle?.dispose();
    };
}
