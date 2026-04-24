import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { WebSocketServer, WebSocket } from "ws";

import { PROTOCOL_VERSION } from "tlmprotocol";
import type { Envelope } from "tlmprotocol";

// ── CLI args ──────────────────────────────────────────────────────────────────

const args = process.argv.slice(2);
let port = 8765;
let host = "127.0.0.1";

for (let i = 0; i < args.length; i++) {
    if (args[i] === "--port" && args[i + 1]) {
        port = parseInt(args[++i]);
    } else if (args[i] === "--host" && args[i + 1]) {
        host = args[++i];
        if (host === "0.0.0.0") {
            console.warn(
                "Warning: binding to 0.0.0.0 exposes the server to all network interfaces",
            );
        }
    }
}

// ── Locate built tlmstudio ────────────────────────────────────────────────────

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const studioDistDir = path.resolve(__dirname, "../../tlmstudio/dist");

const mimeTypes: Record<string, string> = {
    ".html": "text/html",
    ".js":   "application/javascript",
    ".css":  "text/css",
    ".ico":  "image/x-icon",
};

function serveStatic(urlPath: string, res: http.ServerResponse): boolean {
    const filePath = path.join(studioDistDir, urlPath);
    // Prevent path traversal outside studioDistDir
    if (!filePath.startsWith(studioDistDir + path.sep) && filePath !== studioDistDir) {
        return false;
    }
    if (!fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
        return false;
    }
    const ext = path.extname(filePath);
    const contentType = mimeTypes[ext] ?? "application/octet-stream";
    res.writeHead(200, { "Content-Type": contentType });
    fs.createReadStream(filePath).pipe(res);
    return true;
}

// ── WebSocket clients ─────────────────────────────────────────────────────────

const clients = new Set<WebSocket>();

function broadcast(raw: string): void {
    for (const client of clients) {
        if (client.readyState === WebSocket.OPEN) {
            client.send(raw);
        }
    }
}

// ── HTTP server ───────────────────────────────────────────────────────────────

const server = http.createServer((req, res) => {
    const url = req.url ?? "/";

    if (req.method === "POST" && url === "/push") {
        let body = "";
        req.on("data", (chunk) => (body += chunk));
        req.on("end", () => {
            let envelope: Envelope;
            try {
                envelope = JSON.parse(body) as Envelope;
            } catch {
                res.writeHead(400, { "Content-Type": "application/json" });
                res.end(JSON.stringify({ error: "invalid JSON" }));
                return;
            }

            if (envelope.v !== PROTOCOL_VERSION) {
                res.writeHead(400, { "Content-Type": "application/json" });
                res.end(
                    JSON.stringify({
                        error: `expected v=${PROTOCOL_VERSION}, got v=${envelope.v}`,
                    }),
                );
                return;
            }

            if (!envelope.type || !envelope.topic) {
                res.writeHead(400, { "Content-Type": "application/json" });
                res.end(
                    JSON.stringify({
                        error: "missing required fields: type, topic",
                    }),
                );
                return;
            }

            if (clients.size === 0) {
                console.warn(`push  topic=${envelope.topic} type=${envelope.type} — no clients connected, message dropped`);
            } else {
                broadcast(body);
                console.log(`push  topic=${envelope.topic} type=${envelope.type} clients=${clients.size}`);
            }

            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ ok: true }));
        });
        return;
    }

    if (req.method === "GET") {
        if (url === "/") {
            const indexPath = path.join(studioDistDir, "index.html");
            if (!fs.existsSync(indexPath)) {
                res.writeHead(503, { "Content-Type": "text/plain" });
                res.end(`tlmstudio build not found in ${studioDistDir}\nRun: npm run build -w tlmstudio`);
                return;
            }
            res.writeHead(200, { "Content-Type": "text/html" });
            fs.createReadStream(indexPath).pipe(res);
            return;
        }

        if (serveStatic(url, res)) return;
    }

    res.writeHead(404);
    res.end();
});

// ── WebSocket server ──────────────────────────────────────────────────────────

const wss = new WebSocketServer({ server, path: "/ws" });

wss.on("connection", (ws, req) => {
    clients.add(ws);
    const clientAddr = req.socket.remoteAddress ?? "unknown";
    console.log(`ws    connect    client=${clientAddr} total=${clients.size}`);
    ws.on("close", () => {
        clients.delete(ws);
        console.log(`ws    disconnect client=${clientAddr} total=${clients.size}`);
    });
});

// ── Start ─────────────────────────────────────────────────────────────────────

server.listen(port, host, () => {
    const addr = `http://${host}:${port}`;
    console.log(`tlmserver listening on ${addr}`);
    console.log(`  open browser:  ${addr}/`);
    console.log(
        `  push scene:    curl -X POST ${addr}/push -d @scene.json -H 'content-type: application/json'`,
    );
});
