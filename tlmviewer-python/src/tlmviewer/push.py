import json
import urllib.request

from .http import execute_push
from .serialize import scene_to_dict
from .types import Scene

PROTOCOL_VERSION = 2


def push_scene(
    scene: Scene,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    topic: str = "main",
) -> None:
    """Push a Scene to a running tlmserver."""
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

    execute_push(req, url)
