import json
import urllib.error
import urllib.request
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
import tlmviewer as tlmv
from tlmviewer.push import PROTOCOL_VERSION


SIMPLE_SCENE = tlmv.Scene(data=[tlmv.SceneTitle(title="test")])


def _mock_response(body: dict, status: int = 200):
    mock = MagicMock()
    mock.read.return_value = json.dumps(body).encode()
    mock.status = status
    mock.__enter__ = lambda s: s
    mock.__exit__ = MagicMock(return_value=False)
    return mock


# ── Envelope structure ─────────────────────────────────────────────────────────

def test_push_sends_correct_envelope():
    with patch("urllib.request.urlopen", return_value=_mock_response({"ok": True})) as mock_open:
        tlmv.push_scene(SIMPLE_SCENE, topic="mytopic")

    request = mock_open.call_args[0][0]
    envelope = json.loads(request.data)

    assert envelope["v"] == PROTOCOL_VERSION
    assert envelope["type"] == "scene"
    assert envelope["topic"] == "mytopic"
    assert "payload" in envelope
    assert "mode" not in envelope


def test_push_default_topic():
    with patch("urllib.request.urlopen", return_value=_mock_response({"ok": True})) as mock_open:
        tlmv.push_scene(SIMPLE_SCENE)

    envelope = json.loads(mock_open.call_args[0][0].data)
    assert envelope["topic"] == "main"


def test_push_payload_is_scene_dict():
    with patch("urllib.request.urlopen", return_value=_mock_response({"ok": True})) as mock_open:
        tlmv.push_scene(SIMPLE_SCENE)

    envelope = json.loads(mock_open.call_args[0][0].data)
    assert envelope["payload"] == tlmv.scene_to_dict(SIMPLE_SCENE)


# ── Error handling ─────────────────────────────────────────────────────────────

def test_push_raises_value_error_on_http_error():
    error = urllib.error.HTTPError(
        url="http://127.0.0.1:8765/push",
        code=400,
        msg="Bad Request",
        hdrs=None,
        fp=BytesIO(json.dumps({"error": "missing required fields"}).encode()),
    )
    with patch("urllib.request.urlopen", side_effect=error):
        with pytest.raises(ValueError, match="missing required fields"):
            tlmv.push_scene(SIMPLE_SCENE)


def test_push_raises_connection_refused_when_unreachable():
    import socket
    error = urllib.error.URLError(reason=socket.gaierror("connection refused"))
    with patch("urllib.request.urlopen", side_effect=error):
        with pytest.raises(ConnectionRefusedError):
            tlmv.push_scene(SIMPLE_SCENE)


def test_push_raises_runtime_error_on_unexpected_response():
    with patch("urllib.request.urlopen", return_value=_mock_response({"ok": False, "msg": "unexpected"})):
        with pytest.raises(RuntimeError):
            tlmv.push_scene(SIMPLE_SCENE)


# ── TLMVIEWER_PUSH_SCENE_ALLOW_FAIL ───────────────────────────────────────────

def test_allow_fail_silences_url_error(monkeypatch):
    import socket
    monkeypatch.setenv("TLMVIEWER_PUSH_SCENE_ALLOW_FAIL", "1")
    error = urllib.error.URLError(reason=socket.gaierror("connection refused"))
    with patch("urllib.request.urlopen", side_effect=error):
        tlmv.push_scene(SIMPLE_SCENE)  # must not raise


def test_allow_fail_silences_http_error(monkeypatch):
    monkeypatch.setenv("TLMVIEWER_PUSH_SCENE_ALLOW_FAIL", "1")
    error = urllib.error.HTTPError(
        url="http://127.0.0.1:8765/push", code=400, msg="Bad Request",
        hdrs=None, fp=BytesIO(b"{}"),
    )
    with patch("urllib.request.urlopen", side_effect=error):
        tlmv.push_scene(SIMPLE_SCENE)  # must not raise


def test_allow_fail_not_set_still_raises(monkeypatch):
    monkeypatch.delenv("TLMVIEWER_PUSH_SCENE_ALLOW_FAIL", raising=False)
    import socket
    error = urllib.error.URLError(reason=socket.gaierror("connection refused"))
    with patch("urllib.request.urlopen", side_effect=error):
        with pytest.raises(ConnectionRefusedError):
            tlmv.push_scene(SIMPLE_SCENE)
