import json
import os
import urllib.request
import urllib.error

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
    allow_fail = "TLMVIEWER_PUSH_SCENE_ALLOW_FAIL" in os.environ

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

    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
            if not result.get("ok"):
                raise RuntimeError(f"tlmserver returned unexpected response: {result}")
    except urllib.error.HTTPError as e:
        if allow_fail:
            return
        try:
            detail = json.loads(e.read()).get("error", e.reason)
        except Exception:
            detail = e.reason
        raise ValueError(
            f"tlmserver rejected the push (HTTP {e.code}): {detail}"
        ) from e
    except urllib.error.URLError as e:
        if allow_fail:
            return
        raise ConnectionRefusedError(
            f"Could not connect to tlmserver at {url}: {e.reason}"
        ) from e
