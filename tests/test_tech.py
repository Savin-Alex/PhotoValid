import io

from PIL import Image

from backend.validators.tech import TechValidator


def _jpeg(size=(600, 600), color=(120, 60, 30), quality=90):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _tv(size=(600, 600), color=(120, 60, 30), raw=None, ct="image/jpeg", fmt="JPEG"):
    img = Image.new("RGB", size, color)
    if raw is None:
        raw = _jpeg(size, color)
    return TechValidator(img, raw, ct, fmt)


# --- dimensions ---

def test_dimensions_square_in_range_passes():
    assert _tv((600, 600)).dimensions()["status"] == "pass"
    assert _tv((1200, 1200)).dimensions()["status"] == "pass"


def test_dimensions_non_square_fails():
    assert _tv((600, 800)).dimensions()["status"] == "fail"


def test_dimensions_out_of_range_fails():
    assert _tv((500, 500)).dimensions()["status"] == "fail"
    assert _tv((1300, 1300)).dimensions()["status"] == "fail"


# --- file size ---

def test_filesize_over_limit_fails():
    big = b"\x00" * (241 * 1024)  # > 240 KB
    assert _tv(raw=big).filesize()["status"] == "fail"


def test_filesize_small_passes():
    assert _tv(raw=b"\x00" * 1024).filesize()["status"] == "pass"


# --- file format ---

def test_fileformat_jpeg_passes():
    assert _tv(ct="image/jpeg", fmt="JPEG").fileformat()["status"] == "pass"


def test_fileformat_png_fails():
    assert _tv(ct="image/png", fmt="PNG").fileformat()["status"] == "fail"


# --- color model ---

def test_color_model_color_passes():
    assert _tv((600, 600), color=(120, 60, 30)).color_model()["status"] == "pass"


def test_color_model_grayscale_fails():
    assert _tv((600, 600), color=(128, 128, 128)).color_model()["status"] == "fail"


def test_color_model_grayscale_original_mode_fails():
    buf = io.BytesIO()
    Image.new("L", (600, 600), 128).save(buf, format="JPEG")
    raw = buf.getvalue()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    assert TechValidator(img, raw, "image/jpeg", "JPEG").color_model()["status"] == "fail"


def test_color_model_with_srgb_profile_passes():
    from PIL import ImageCms

    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    buf = io.BytesIO()
    Image.new("RGB", (600, 600), (120, 60, 30)).save(buf, format="JPEG", icc_profile=icc)
    raw = buf.getvalue()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    assert TechValidator(img, raw, "image/jpeg", "JPEG").color_model()["status"] == "pass"


# --- compression ratio (official: <= 20:1) ---

def test_compression_ratio_overcompressed_fails():
    # 600x600x3 = 1,080,000 uncompressed; 20 KB file -> ratio ~54:1
    res = _tv((600, 600), raw=b"\x00" * 20_000).compression_ratio()
    assert res["status"] == "fail"
    assert res["expected"] == "≤ 20:1"


def test_compression_ratio_normal_passes():
    # 1,080,000 / 100,000 = 10.8:1 -> within spec
    res = _tv((600, 600), raw=b"\x00" * 100_000).compression_ratio()
    assert res["status"] == "pass"


# --- brightness / contrast ---

def test_brightness_contrast_returns_two_structured_results():
    out = _tv((600, 600), color=(120, 60, 30)).brightness_contrast()
    assert len(out) == 2
    for r in out:
        assert r["status"] in {"pass", "warning", "fail", "skipped"}
        assert r["name"] in {"Brightness", "Contrast"}


def test_brightness_measured_on_center_not_whole_image():
    # White background with a well-exposed mid-gray subject in the center.
    # Whole-image average would read ~90% (too bright); the center is ~59%.
    import numpy as np
    arr = np.full((600, 600, 3), 255, dtype=np.uint8)
    arr[120:480, 150:450] = 150
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    out = TechValidator(img, buf.getvalue(), "image/jpeg", "JPEG").brightness_contrast()
    brightness = next(r for r in out if r["name"] == "Brightness")
    assert brightness["status"] == "pass"


# --- run() shape ---

def test_run_includes_compression_ratio():
    names = [r["name"] for r in _tv().run()]
    assert "Compression Ratio" in names


def test_exif_age_reads_date_from_raw_bytes():
    import piexif
    from datetime import datetime

    exif_bytes = piexif.dump(
        {"Exif": {piexif.ExifIFD.DateTimeOriginal: datetime.now().strftime("%Y:%m:%d %H:%M:%S")}}
    )
    buf = io.BytesIO()
    Image.new("RGB", (600, 600), (120, 60, 30)).save(buf, format="JPEG", exif=exif_bytes)
    raw = buf.getvalue()
    # The converted PIL image may have dropped EXIF, but the raw bytes retain it.
    img = Image.open(io.BytesIO(raw)).convert("RGB")

    res = TechValidator(img, raw, "image/jpeg", "JPEG").exif_age()

    assert res["status"] == "pass"
    assert "Unknown" not in str(res["value"])
