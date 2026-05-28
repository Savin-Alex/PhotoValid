from backend.main import _summarize_response
from backend.validators.utils import json_param


def test_skipped_checks_excluded_from_score_and_status():
    technical = [json_param("File Size", "10 KB", "<= 240 KB", True)]
    biometric = [json_param("Red-Eye", "manual", "No red-eye", False, status="skipped")]

    out = _summarize_response(technical, biometric, [])

    # The skipped (manual-review) check must not drag the score or block a pass.
    assert out["overall_score"] == 100
    assert out["status"] == "pass"
    assert out["ok"] is True
    assert out["manual_review"]  # surfaced separately


def test_critical_fail_is_fail():
    technical = [json_param("Photo Dimensions", "800x600", "square", False)]

    out = _summarize_response(technical, [], [])

    assert out["status"] == "fail"
    assert any("Photo Dimensions" in e for e in out["errors"])


def test_non_critical_fail_is_only_warning():
    technical = [json_param("File Size", "10 KB", "<= 240 KB", True)]
    biometric = [json_param("Head Centering", "9% offset", "<= 5%", False)]  # advisory

    out = _summarize_response(technical, biometric, [])

    assert out["status"] == "warning"
    assert any("Head Centering" in w for w in out["warnings"])
    assert not out["errors"]


def test_warning_status_for_warned_check():
    technical = [json_param("Brightness", "70%", "45-65%", False, warn=True)]

    out = _summarize_response(technical, [], [])

    assert out["status"] == "warning"


def test_top_level_error_is_fatal():
    out = _summarize_response([], [], [], errors=["Uploaded file is not a valid supported image."])

    assert out["status"] == "fail"
    assert out["ok"] is False


def test_critical_skipped_forces_warning_not_pass():
    # A required (critical) check that couldn't be verified must NOT look like a pass.
    technical = [json_param("File Size", "10 KB", "<= 240 KB", True)]
    biometric = [json_param("Face Detection", "Not checked", "One face detected", False,
                            status="skipped", rec="Face analysis could not run.")]

    out = _summarize_response(technical, biometric, [])

    assert out["status"] == "warning"
    assert out["ok"] is False
    fd = next(c for c in out["checks"] if c["name"] == "Face Detection")
    assert fd["critical"] is True
    assert fd["skipped_reason"] == "Face analysis could not run."
    assert any("Face Detection" in w for w in out["warnings"])
    assert not out["errors"]


def test_noncritical_skip_does_not_block_pass():
    technical = [json_param("File Size", "10 KB", "<= 240 KB", True)]
    biometric = [json_param("Red-Eye", "Not auto-checked", "No red-eye", False, status="skipped")]

    out = _summarize_response(technical, biometric, [])

    assert out["status"] == "pass"
    rec = next(c for c in out["checks"] if c["name"] == "Red-Eye")
    assert rec["critical"] is False
    assert out["manual_review"]


def test_all_pass_is_pass():
    technical = [
        json_param("Photo Dimensions", "600x600", "square", True),
        json_param("File Size", "100 KB", "<= 240 KB", True),
    ]

    out = _summarize_response(technical, [], [])

    assert out["status"] == "pass"
    assert out["overall_score"] == 100
