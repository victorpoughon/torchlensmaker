import json
import logging
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)


def execute_push(req: urllib.request.Request, url: str) -> None:
    """Execute a push request to tlmserver.

    Raises on server errors (HTTPError, unexpected response).
    Logs a warning and returns silently if the server is not reachable (URLError).
    """
    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
            if not result.get("ok"):
                raise RuntimeError(f"tlmserver returned unexpected response: {result}")
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read()).get("error", e.reason)
        except Exception:
            detail = e.reason
        raise ValueError(
            f"tlmserver rejected the push (HTTP {e.code}): {detail}"
        ) from e
    except urllib.error.URLError as e:
        logger.warning("Could not connect to tlmserver at %s: %s", url, e.reason)
