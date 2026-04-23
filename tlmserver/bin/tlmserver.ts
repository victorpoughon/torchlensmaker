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

// ── Locate built tlmviewer ────────────────────────────────────────────────────

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const distDir = path.resolve(__dirname, "../../tlmviewer/dist");

function findTlmviewerUmd(): string | null {
    try {
        const files = fs.readdirSync(distDir);
        const umd = files.find((f) => f.endsWith(".umd.js"));
        return umd ? path.join(distDir, umd) : null;
    } catch {
        return null;
    }
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

// ── HTML page ─────────────────────────────────────────────────────────────────

const indexHtml = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>tlmserver</title>
  <style>body { margin: 0; overflow: hidden; }</style>
</head>
<body>
  <div id="viewer" class="tlmviewer" style="width:100vw;height:100vh;"></div>
  <script src="/tlmviewer.js"></script>
  <script>
    const wsUrl = "ws://" + location.host + "/ws";
    tlmviewer.connect(document.getElementById("viewer"), wsUrl);
  </script>
</body>
</html>`;

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

    if (req.method === "GET" && url === "/") {
        res.writeHead(200, { "Content-Type": "text/html" });
        res.end(indexHtml);
        return;
    }

    if (req.method === "GET" && url === "/tlmviewer.js") {
        const umdPath = findTlmviewerUmd();
        if (!umdPath) {
            res.writeHead(503, { "Content-Type": "text/plain" });
            res.end(
                `tlmviewer UMD build not found in ${distDir}\nRun: npm run build -w tlmviewer`,
            );
            return;
        }
        res.writeHead(200, { "Content-Type": "application/javascript" });
        fs.createReadStream(umdPath).pipe(res);
        return;
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
