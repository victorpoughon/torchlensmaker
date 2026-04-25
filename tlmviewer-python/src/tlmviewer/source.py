import json
import os
import urllib.request

from .http import execute_push

PROTOCOL_VERSION = 2

_LANGUAGE_MAP = {
    ".py": "python",
    ".ts": "typescript",
    ".js": "javascript",
}


def push_source(
    filepath: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    topic: str = "main",
) -> None:
    """Push a source file to a running tlmserver to display in tlmstudio."""
    filepath = os.path.abspath(filepath)
    filename = os.path.basename(filepath)
    ext = os.path.splitext(filename)[1].lower()
    language = _LANGUAGE_MAP.get(ext, "text")

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    envelope = {
        "v": PROTOCOL_VERSION,
        "type": "source",
        "topic": topic,
        "payload": {
            "filename": filename,
            "language": language,
            "content": content,
        },
    }

    body = json.dumps(envelope).encode()
    url = f"http://{host}:{port}/push"
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    execute_push(req, url)
