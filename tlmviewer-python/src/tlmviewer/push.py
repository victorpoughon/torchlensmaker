import json
import os
import urllib.request
import urllib.error

from .types import Scene
from .serialize import scene_to_dict

PROTOCOL_VERSION = 2


def push_scene(
    scene: Scene,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    topic: str = "main",
) -> None:
    """Push a Scene to a running tlmserver.

    If the environment variable TLMVIEWER_PUSH_SCENE_ALLOW_FAIL is set,
    connection errors are silently ignored instead of raised.
    """
    allow_fail = "TLMVIEWER_PUSH_SCENE_ALLOW_FAIL" in os.environ

    envelope = {
        "v": PROTOCOL_VERSION,
        "type": "scene",
        "topic": topic,
        "payload": scene_to_dict(scene),
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
