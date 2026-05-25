import numpy as np
from PIL import Image

from backend.validators.tamper import TamperValidator


def _img(color=(120, 60, 30), size=(256, 256)):
    return Image.new("RGB", size, color)


def test_run_returns_four_structured_checks():
    results = TamperValidator(_img()).run()
    assert len(results) == 4
    for r in results:
        assert r["status"] in {"pass", "warning", "fail", "skipped"}
        assert r["name"]


def test_metadata_consistency_missing_exif_is_not_pass():
    # A freshly created PIL image carries no EXIF metadata.
    res = TamperValidator(_img()).metadata_consistency()
    assert res["status"] != "pass"


def test_error_level_analysis_is_structured():
    res = TamperValidator(_img()).error_level_analysis()
    assert res["name"] == "Error Level Analysis"
    assert res["status"] in {"pass", "warning", "fail", "skipped"}


def test_image_noise_clean_is_low():
    res = TamperValidator(_img()).image_noise()  # solid color -> flat -> low noise
    assert res["name"] == "Image Noise"
    assert res["status"] == "pass"


def test_image_noise_noisy_is_flagged():
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    res = TamperValidator(Image.fromarray(arr)).image_noise()
    assert res["status"] in {"warning", "fail"}


def test_color_channel_consistency_is_structured():
    res = TamperValidator(_img()).color_channel_consistency()
    assert res["name"] == "Color Channel Balance"
    assert res["status"] in {"pass", "warning", "fail", "skipped"}
