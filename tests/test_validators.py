import io

import numpy as np
import pytest
from PIL import Image

import backend.validators.bio as bio_module
from backend.validators.bio import BioValidator
from backend.validators.tech import TechValidator
from backend.validators.utils import json_param


def test_tech_photo_date_without_exif_does_not_silently_pass():
    """A photo with no EXIF capture date must NOT be reported as recent.

    Regression: the old fallback read the temp file's mtime (= now), so any
    photo lacking a capture date always 'passed' the 6-month recency check.
    """
    buf = io.BytesIO()
    Image.new("RGB", (600, 600), "white").save(buf, format="JPEG")  # no EXIF date
    raw = buf.getvalue()
    img = Image.open(io.BytesIO(raw))

    result = TechValidator(img, raw, "image/jpeg", "JPEG").exif_age()

    assert result["name"] == "Photo Date"
    assert result["status"] != "pass"


def test_bio_no_face_returns_failed_or_skipped_detection():
    arr = np.full((600, 600, 3), 255, dtype=np.uint8)

    results = BioValidator(arr).run()

    detection = next(item for item in results if item["name"] == "Face Detection")
    assert detection["status"] in {"fail", "skipped"}


def test_bio_multiple_faces_fails_one_person(monkeypatch):
    arr = np.full((600, 600, 3), 255, dtype=np.uint8)

    def fake_calculate(self, manual_overrides=None):
        self.face_count = 2
        face_box = {
            "top": 120,
            "bottom": 480,
            "eyeY": 250,
            "left": 180,
            "right": 420,
            "centerX": 300,
            "image_height": 600,
            "image_width": 600,
            "method": "facemesh",
        }
        return {
            "faceBox": face_box,
            "head_ratio": 60.0,
            "eye_level": 58.3,
            "center_offset": 0.0,
        }

    monkeypatch.setattr(bio_module, "MP_AVAILABLE", True)
    monkeypatch.setattr(BioValidator, "calculate", fake_calculate)
    monkeypatch.setattr(
        BioValidator,
        "check_background",
        lambda self: json_param("Background", "Plain", "Plain white/off-white", True),
    )
    monkeypatch.setattr(
        BioValidator,
        "check_sharpness",
        lambda self, face_box: json_param("Sharpness", "Sharp", "Sharp focus", True),
    )
    monkeypatch.setattr(
        BioValidator,
        "check_lighting",
        lambda self, face_box: json_param("Face Lighting", "Even", "Even lighting", True),
    )

    results = BioValidator(arr).run()

    one_person = next(item for item in results if item["name"] == "One Person Only")
    assert one_person["status"] == "fail"


def test_bio_unverifiable_checks_do_not_auto_pass(monkeypatch):
    """Eyeglasses/headgear/expression/red-eye are not auto-detected and must be
    reported as 'skipped' (manual review), never as a silent 'pass'."""
    arr = np.full((600, 600, 3), 255, dtype=np.uint8)

    def fake_calculate(self, manual_overrides=None):
        self.face_count = 1
        face_box = {
            "top": 120,
            "bottom": 480,
            "eyeY": 250,
            "left": 180,
            "right": 420,
            "centerX": 300,
            "image_height": 600,
            "image_width": 600,
            "method": "facemesh",
        }
        return {
            "faceBox": face_box,
            "head_ratio": 60.0,
            "eye_level": 58.3,
            "center_offset": 0.0,
        }

    monkeypatch.setattr(bio_module, "MP_AVAILABLE", True)
    monkeypatch.setattr(BioValidator, "calculate", fake_calculate)
    monkeypatch.setattr(
        BioValidator,
        "check_background",
        lambda self: json_param("Background", "Plain", "Plain white/off-white", True),
    )
    monkeypatch.setattr(
        BioValidator,
        "check_sharpness",
        lambda self, face_box: json_param("Sharpness", "Sharp", "Sharp focus", True),
    )
    monkeypatch.setattr(
        BioValidator,
        "check_lighting",
        lambda self, face_box: json_param("Face Lighting", "Even", "Even lighting", True),
    )

    results = BioValidator(arr).run()
    by_name = {item["name"]: item for item in results}

    for name in ("Glasses/Headphones", "Headgear", "Facial Expression", "Red-Eye"):
        assert by_name[name]["status"] == "skipped", name
        assert by_name[name]["status"] != "pass", name


def _fake_landmarks(eye_y=0.45):
    """Minimal stand-in for a MediaPipe FaceMesh landmark list.

    Only the indices used by check_glasses (133, 362, 168) are positioned;
    the rest default to image center.
    """
    class _Pt:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.z = 0.0

    class _LMS:
        def __init__(self, pts):
            self.landmark = pts

    pts = [_Pt(0.5, 0.5) for _ in range(478)]
    pts[133] = _Pt(0.42, eye_y)   # left inner eye corner
    pts[362] = _Pt(0.58, eye_y)   # right inner eye corner
    pts[168] = _Pt(0.50, eye_y)   # nose bridge
    return _LMS(pts)


def test_check_glasses_without_landmarks_is_skipped():
    bio = BioValidator(np.full((400, 400, 3), 200, dtype=np.uint8))  # landmarks is None
    res = bio.check_glasses()
    assert res["name"] == "Glasses/Headphones"
    assert res["status"] == "skipped"


def test_check_glasses_smooth_bridge_is_skipped():
    bio = BioValidator(np.full((400, 400, 3), 200, dtype=np.uint8))
    bio.landmarks = _fake_landmarks()
    res = bio.check_glasses()
    # A smooth nose bridge must NOT be confidently flagged as glasses.
    assert res["status"] == "skipped"


def test_check_glasses_strong_bridge_edges_are_flagged():
    cv2 = pytest.importorskip("cv2")
    arr = np.full((400, 400, 3), 200, dtype=np.uint8)
    # Hatch the nose-bridge region with strong edges (mimics a glasses frame).
    for yy in range(168, 196, 4):
        cv2.line(arr, (150, yy), (250, yy), (10, 10, 10), 1)
    bio = BioValidator(arr)
    bio.landmarks = _fake_landmarks()
    res = bio.check_glasses()
    assert res["status"] in {"warning", "fail"}


def test_check_redeye_without_landmarks_is_skipped():
    bio = BioValidator(np.full((400, 400, 3), 180, dtype=np.uint8))  # landmarks None
    res = bio.check_redeye()
    assert res["name"] == "Red-Eye"
    assert res["status"] == "skipped"


def test_check_redeye_no_red_is_skipped():
    arr = np.full((400, 400, 3), 180, dtype=np.uint8)  # neutral gray, no red
    bio = BioValidator(arr)
    bio.landmarks = _mesh({468: (0.40, 0.45), 473: (0.60, 0.45)})
    res = bio.check_redeye()
    assert res["status"] == "skipped"


def test_check_redeye_strong_red_in_iris_is_flagged():
    cv2 = pytest.importorskip("cv2")
    arr = np.full((400, 400, 3), 180, dtype=np.uint8)
    # Paint pure red (BGR) at the iris locations (468->~160,180; 473->~240,180).
    cv2.circle(arr, (160, 180), 8, (0, 0, 255), -1)
    cv2.circle(arr, (240, 180), 8, (0, 0, 255), -1)
    bio = BioValidator(arr)
    bio.landmarks = _mesh({468: (0.40, 0.45), 473: (0.60, 0.45)})
    res = bio.check_redeye()
    assert res["status"] in {"warning", "fail"}


def _mesh(points):
    """Synthetic FaceMesh landmark list; `points` maps index -> (x, y) normalized."""
    class _Pt:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.z = 0.0

    class _LMS:
        def __init__(self, pts):
            self.landmark = pts

    pts = [_Pt(0.5, 0.5) for _ in range(478)]
    for i, (x, y) in points.items():
        pts[i] = _Pt(x, y)
    return _LMS(pts)


def _bio_with(points):
    bio = BioValidator(np.zeros((400, 400, 3), dtype=np.uint8))
    bio.landmarks = _mesh(points)
    return bio


# --- Eyes open (EAR) ---

def test_check_eyes_open_detects_open_eyes():
    res = _bio_with({
        33: (0.30, 0.45), 133: (0.40, 0.45),
        160: (0.35, 0.42), 158: (0.37, 0.43), 153: (0.37, 0.47), 144: (0.35, 0.48),
        263: (0.70, 0.45), 362: (0.60, 0.45),
        385: (0.65, 0.42), 387: (0.63, 0.43), 373: (0.63, 0.47), 380: (0.65, 0.48),
    }).check_eyes_open()
    assert res["name"] == "Eyes Open"
    assert res["status"] == "pass"


def test_check_eyes_open_detects_closed_eyes():
    # All eyelid points share a y -> vertical gap ~0 -> EAR ~0.
    res = _bio_with({
        33: (0.30, 0.45), 133: (0.40, 0.45),
        160: (0.35, 0.45), 158: (0.37, 0.45), 153: (0.37, 0.45), 144: (0.35, 0.45),
        263: (0.70, 0.45), 362: (0.60, 0.45),
        385: (0.65, 0.45), 387: (0.63, 0.45), 373: (0.63, 0.45), 380: (0.65, 0.45),
    }).check_eyes_open()
    assert res["status"] in {"fail", "warning"}


# --- Gaze (iris centering) ---

def test_check_gaze_centered_passes():
    res = _bio_with({
        33: (0.30, 0.45), 133: (0.40, 0.45), 468: (0.35, 0.45),
        263: (0.70, 0.45), 362: (0.60, 0.45), 473: (0.65, 0.45),
    }).check_gaze()
    assert res["status"] == "pass"


def test_check_gaze_off_center_is_flagged():
    res = _bio_with({
        33: (0.30, 0.45), 133: (0.40, 0.45), 468: (0.385, 0.45),
        263: (0.70, 0.45), 362: (0.60, 0.45), 473: (0.685, 0.45),
    }).check_gaze()
    assert res["status"] in {"warning", "fail"}


# --- Expression (mouth openness) ---

def test_check_expression_closed_mouth_passes():
    res = _bio_with({
        61: (0.40, 0.70), 291: (0.60, 0.70), 13: (0.50, 0.69), 14: (0.50, 0.70),
    }).check_expression()
    assert res["name"] == "Facial Expression"
    assert res["status"] == "pass"


def test_check_expression_open_mouth_is_flagged():
    res = _bio_with({
        61: (0.40, 0.70), 291: (0.60, 0.70), 13: (0.50, 0.62), 14: (0.50, 0.78),
    }).check_expression()
    assert res["status"] in {"warning", "fail"}


# --- Head tilt (roll) ---

def test_check_head_tilt_level_passes():
    res = _bio_with({468: (0.40, 0.45), 473: (0.60, 0.45)}).check_head_tilt()
    assert res["name"] == "Head Tilt"
    assert res["status"] == "pass"


def test_check_head_tilt_tilted_is_flagged():
    # eyes on a clearly slanted line (~27 degrees)
    res = _bio_with({468: (0.40, 0.40), 473: (0.60, 0.50)}).check_head_tilt()
    assert res["status"] in {"warning", "fail"}


def test_geometry_checks_without_landmarks_are_skipped():
    bio = BioValidator(np.zeros((400, 400, 3), dtype=np.uint8))  # landmarks None
    assert bio.check_eyes_open()["status"] == "skipped"
    assert bio.check_gaze()["status"] == "skipped"
    assert bio.check_expression()["status"] == "skipped"
    assert bio.check_head_tilt()["status"] == "skipped"


# --- Background (top-corner sampling) ---

def test_check_background_clean_white_passes():
    pytest.importorskip("cv2")
    arr = np.full((600, 600, 3), 255, dtype=np.uint8)  # plain white corners
    res = BioValidator(arr).check_background()
    assert res["name"] == "Background"
    assert res["status"] == "pass"


def test_check_background_strong_color_not_pass():
    pytest.importorskip("cv2")
    arr = np.zeros((600, 600, 3), dtype=np.uint8)
    arr[:] = (200, 40, 40)  # strongly colored (BGR) -> non-neutral background
    res = BioValidator(arr).check_background()
    assert res["status"] != "pass"


def test_check_background_noisy_not_pass():
    pytest.importorskip("cv2")
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (600, 600, 3), dtype=np.uint8)  # busy corners
    res = BioValidator(arr).check_background()
    assert res["status"] != "pass"
