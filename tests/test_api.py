import io

from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app


client = TestClient(app)


def _jpeg_bytes(size=(600, 600), color="white"):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def test_validate_rejects_corrupted_image_with_canonical_shape():
    response = client.post(
        "/api/validate",
        files={"file": ("broken.jpg", b"not an image", "image/jpeg")},
    )

    payload = response.json()
    assert response.status_code == 400
    assert payload["ok"] is False
    assert payload["status"] == "fail"
    assert payload["errors"]
    assert payload["technical"] == []
    assert payload["biometric"] == []
    assert payload["tamper"] == []
    assert payload["checks"] == []


def test_validate_returns_canonical_shape_for_valid_jpeg():
    response = client.post(
        "/api/validate",
        files={"file": ("photo.jpg", _jpeg_bytes(), "image/jpeg")},
    )

    payload = response.json()
    assert response.status_code == 200
    assert {"ok", "status", "overall_score", "errors", "warnings", "technical", "biometric", "tamper", "checks"} <= payload.keys()
    assert payload["technical"]
    assert isinstance(payload["checks"], list)


def test_validate_accepts_manual_overrides():
    response = client.post(
        "/api/validate",
        files={"file": ("photo.jpg", _jpeg_bytes(), "image/jpeg")},
        data={"overrides": '{"top": 0.18, "eye": 0.40, "chin": 0.86}'},
    )

    payload = response.json()
    assert response.status_code == 200
    names = [c.get("name") for c in payload["checks"]]
    # Manual mode produces geometry-based checks without needing a real face.
    assert "Head Height" in names
    assert "Eye Level" in names


def test_validate_invalid_overrides_json_is_warned_not_fatal():
    response = client.post(
        "/api/validate",
        files={"file": ("photo.jpg", _jpeg_bytes(), "image/jpeg")},
        data={"overrides": "not-json"},
    )

    payload = response.json()
    assert response.status_code == 200
    assert any("override" in w.lower() for w in payload["warnings"])


def test_validate_includes_compression_ratio_check():
    response = client.post(
        "/api/validate",
        files={"file": ("photo.jpg", _jpeg_bytes(), "image/jpeg")},
    )

    payload = response.json()
    names = [c.get("name") for c in payload["technical"]]
    assert "Compression Ratio" in names


def test_report_returns_pdf():
    response = client.post(
        "/api/report",
        files={"file": ("photo.jpg", _jpeg_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content[:5] == b"%PDF-"
    assert len(response.content) > 1000  # a real document, not an empty stub


def test_healthz_is_lightweight():
    response = client.get("/healthz")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    for key in ("version", "python", "opencv_available", "mediapipe_available"):
        assert key in body
    # Liveness must NOT report (or trigger) model init.
    assert "face_detection_ready" not in body


def test_readyz_reports_model_status():
    response = client.get("/readyz")

    assert response.status_code == 200
    body = response.json()
    for key in ("ready", "opencv_available", "mediapipe_available",
                "face_detection_ready", "face_mesh_ready"):
        assert key in body


def test_report_rejects_invalid_image():
    response = client.post(
        "/api/report",
        files={"file": ("broken.jpg", b"not an image", "image/jpeg")},
    )

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
