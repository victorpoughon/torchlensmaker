import json
import pytest
import tlmviewer as tlmv

from conftest import ID_MATRIX, ELEMENTS, ELEMENT_TYPE_STRINGS


# ── Element type strings ───────────────────────────────────────────────────────

@pytest.mark.parametrize("element,expected_type", [(e, ELEMENT_TYPE_STRINGS[type(e)]) for e in ELEMENTS])
def test_element_type_string(element, expected_type):
    d = tlmv.scene_to_dict(tlmv.Scene(data=[element]))
    assert d["data"][0]["type"] == expected_type


# ── Field renames ──────────────────────────────────────────────────────────────

def test_clip_planes_serialized():
    element = tlmv.SurfaceDisk(radius=5.0, matrix=ID_MATRIX, clip_planes=[(0, 1, 0, 0)])
    d = tlmv.scene_to_dict(tlmv.Scene(data=[element]))
    ed = d["data"][0]
    assert list(ed["clip_planes"][0]) == [0, 1, 0, 0]


def test_bspline_periodic_clamped_serialized():
    element = tlmv.SurfaceBSpline(
        points=[[[0, 0, 0]]], weights=[[1.0]], degree=(2, 2),
        periodic=(False, False), clamped=(True, True),
        samples=(10, 10), matrix=ID_MATRIX,
    )
    d = tlmv.scene_to_dict(tlmv.Scene(data=[element]))
    ed = d["data"][0]
    assert ed["periodic"] == (False, False)
    assert ed["clamped"] == (True, True)


def test_sag_function_serialized():
    sag = {"sag-type": "spherical", "C": 0.1}
    element = tlmv.SurfaceSag(diameter=5.0, sag_function=sag, matrix=ID_MATRIX)
    d = tlmv.scene_to_dict(tlmv.Scene(data=[element]))
    assert d["data"][0]["sag_function"] == sag


def test_sag_function_interior_keys_unchanged():
    sag = {"sag-type": "aspheric", "C": 0.1, "K": 0.0, "coefficients": [1, 2, 3]}
    element = tlmv.SurfaceSag(diameter=5.0, sag_function=sag, matrix=ID_MATRIX)
    d = tlmv.scene_to_dict(tlmv.Scene(data=[element]))
    assert d["data"][0]["sag_function"] == sag


# ── Scene optional fields ──────────────────────────────────────────────────────

def test_scene_camera_omitted_when_none():
    d = tlmv.scene_to_dict(tlmv.Scene())
    assert "camera" not in d


def test_scene_camera_included_when_set():
    d = tlmv.scene_to_dict(tlmv.Scene(camera="orthographic"))
    assert d["camera"] == "orthographic"


def test_scene_controls_omitted_when_empty():
    d = tlmv.scene_to_dict(tlmv.Scene())
    assert "controls" not in d


def test_scene_controls_included_when_set():
    d = tlmv.scene_to_dict(tlmv.Scene(controls={"show_axis_x": True}))
    assert d["controls"] == {"show_axis_x": True}


def test_scene_mode():
    assert tlmv.scene_to_dict(tlmv.Scene(mode="2D"))["mode"] == "2D"
    assert tlmv.scene_to_dict(tlmv.Scene(mode="3D"))["mode"] == "3D"


# ── scene_to_json ──────────────────────────────────────────────────────────────

def test_scene_to_json_is_valid_json():
    scene = tlmv.Scene(data=[tlmv.SceneTitle(title="hello")])
    s = tlmv.scene_to_json(scene)
    parsed = json.loads(s)
    assert parsed["data"][0]["title"] == "hello"


@pytest.mark.parametrize("element", ELEMENTS)
def test_scene_to_json_all_element_types(element):
    scene = tlmv.Scene(data=[element])
    json.loads(tlmv.scene_to_json(scene))


# ── Empty clip_planes not omitted (it's always present) ───────────────────────

def test_empty_clip_planes_serialized():
    element = tlmv.SurfaceDisk(radius=5.0, matrix=ID_MATRIX)
    d = tlmv.scene_to_dict(tlmv.Scene(data=[element]))
    assert d["data"][0]["clip_planes"] == []
