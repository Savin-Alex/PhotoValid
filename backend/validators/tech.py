from __future__ import annotations
import io
import math
from typing import Dict, Any
from PIL import Image
import numpy as np
from .utils import is_grayscale, brightness, contrast, exif_capture_datetime, months_between, json_param

# Sharpness is measured on the face region in bio.py (Laplacian variance). The old
# full-image "normalized" score lived here but was miscalibrated (its threshold was
# effectively unreachable, so sharp photos failed) and duplicated the name "Sharpness";
# it has been removed in favor of the single face-ROI check.

MAX_KB = 240


def _icc_is_srgb(icc_bytes: bytes) -> bool:
    """Best-effort check that an embedded ICC profile is sRGB.

    Returns True when we cannot introspect the profile, so a photo is never
    failed merely because its profile could not be parsed.
    """
    try:
        from PIL import ImageCms
        prof = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
        desc = (ImageCms.getProfileDescription(prof) or "")
        return "srgb" in desc.lower()
    except Exception:
        return True


class TechValidator:
    def __init__(self, pil_img: Image.Image, raw_bytes: bytes, content_type: str, detected_format: str | None = None):
        self.img = pil_img
        self.raw = raw_bytes
        self.content_type = content_type
        self.detected_format = (detected_format or getattr(pil_img, "format", None) or "").upper()
        self.arr = np.asarray(pil_img.convert('RGB'))[:,:,::-1].copy()  # BGR for cv

    def dimensions(self) -> Dict[str, Any]:
        w, h = self.img.size
        is_square = (w == h)
        in_range = (600 <= w <= 1200) and (600 <= h <= 1200)
        ok = is_square and in_range
        rec = None
        fix = None
        if not is_square:
            rec = 'Photo must be square (equal width and height).'
            fix = 'Crop your photo to make it square (same width and height). Use photo editing software or online tools.'
        elif not in_range:
            rec = 'Resize to between 600×600 and 1200×1200 px.'
            fix = 'Resize your photo to between 600×600 and 1200×1200 pixels. Most photo editors can do this easily.'
        return json_param('Photo Dimensions', f'{w}×{h}px', '600×600–1200×1200 px', ok, rec=rec, fix=fix)

    def filesize(self) -> Dict[str, Any]:
        kb = math.ceil(len(self.raw)/1024)
        return json_param('File Size', f'{kb} KB', '≤ 240 KB', kb <= MAX_KB,
                         rec='Reduce file size below 240 KB',
                         fix='Compress your photo or reduce quality in photo editing software. Try saving at 85% JPEG quality.')

    def compression_ratio(self) -> Dict[str, Any]:
        # Official DV digital-image spec: compression ratio must be <= 20:1.
        # Ratio = uncompressed size / compressed (file) size. For a 24-bit color
        # image the uncompressed size is width * height * 3 bytes. A high ratio
        # means the JPEG was saved at low quality (visible artifacts).
        w, h = self.img.size
        uncompressed = w * h * 3
        compressed = max(len(self.raw), 1)
        ratio = uncompressed / compressed
        ok = ratio <= 20.0
        return json_param(
            'Compression Ratio',
            f'{ratio:.1f}:1',
            '≤ 20:1',
            ok,
            warn=20.0 < ratio <= 25.0,
            rec='Image is over-compressed (compression ratio above 20:1), which can cause visible artifacts.',
            fix='Re-save the photo at higher JPEG quality (e.g. 85–95%) so it is less compressed, while keeping the file ≤ 240 KB.'
        )

    def fileformat(self) -> Dict[str, Any]:
        declared_jpeg = self.content_type in ('image/jpeg','image/jpg')
        detected_jpeg = self.detected_format in ("JPEG", "JPG")
        ok = declared_jpeg and detected_jpeg
        value = f"{self.content_type or 'unknown'} / {self.detected_format or 'unknown'}"
        return json_param('File Format', value, 'JPEG', ok, 
                         rec='Save as JPEG (.jpg)',
                         fix='Convert your photo to JPEG format. Use "Save As" and select JPEG (.jpg) in your photo editor.')

    def color_model(self) -> Dict[str, Any]:
        grayscale = is_grayscale(self.arr)

        # Inspect the ORIGINAL upload, not the converted copy. main.py converts
        # to RGB before handing us the image, which would hide the true mode and
        # any embedded ICC profile.
        orig_mode = None
        icc = None
        try:
            with Image.open(io.BytesIO(self.raw)) as orig:
                orig_mode = orig.mode
                icc = orig.info.get("icc_profile")
        except Exception:
            pass

        if grayscale:
            return json_param('Color Model', 'Grayscale', 'Color, 24-bit sRGB', False,
                             rec='Use a color photo (no grayscale).',
                             fix='Take a new color photo or convert your black-and-white photo to color using photo editing software.')

        # 24-bit RGB == 8 bits/channel x 3 channels. "L"/"1"/"LA" are grayscale;
        # "P"/"CMYK"/"RGBA" are not the required 24-bit RGB.
        is_24bit_rgb = (orig_mode == "RGB") or (orig_mode is None)

        # sRGB: camera JPEGs usually carry no embedded profile and are sRGB by
        # convention; only flag when a profile is present AND clearly not sRGB.
        if icc:
            srgb_ok = _icc_is_srgb(icc)
            profile_note = "sRGB profile" if srgb_ok else "non-sRGB profile"
        else:
            srgb_ok = True
            profile_note = "no profile (assumed sRGB)"

        ok = is_24bit_rgb and srgb_ok
        value = f"{orig_mode or 'RGB'}, {profile_note}"
        return json_param('Color Model', value, 'Color, 24-bit sRGB', ok, warn=not ok,
                         rec='Photo should be 24-bit color in the sRGB color space.',
                         fix='Re-export as a standard sRGB JPEG (24-bit color) from your photo editor.')

    def _center_region(self) -> np.ndarray:
        # The subject is centered and fills the middle of a DV photo, so a central
        # crop approximates the FACE exposure. Measuring the whole image instead
        # lets the plain (usually white) background dominate the average, making a
        # well-exposed face read as "too bright".
        h, w = self.arr.shape[:2]
        x1, x2 = int(w * 0.25), int(w * 0.75)
        y1, y2 = int(h * 0.20), int(h * 0.80)
        roi = self.arr[y1:y2, x1:x2]
        return roi if roi.size else self.arr

    def brightness_contrast(self) -> list[Dict[str,Any]]:
        roi = self._center_region()
        b = brightness(roi)
        c = contrast(roi)
        out = []

        # Measured on the central (face) region. The DV spec doesn't quantify
        # brightness, so this is a lenient sanity band, not an official threshold.
        out.append(json_param('Brightness', f'{b:.0f}%', '40–72% (face area)', 40 <= b <= 72, warn=30 <= b <= 80,
                             rec='Adjust lighting; avoid over/underexposing the face.',
                             fix='Retake photo with even, natural lighting on the face. Avoid harsh flash or backlight.'))

        # Enhanced contrast with ratio test (also on the central region)
        gray = roi.mean(axis=2) if len(roi.shape) == 3 else roi
        min_val, max_val = float(gray.min()), float(gray.max())
        contrast_ratio = (max_val - min_val) / max_val if max_val > 0 else 0

        ratio_ok = contrast_ratio > 0.5
        out.append(json_param('Contrast', f'{c:.1f}% (ratio: {contrast_ratio:.2f})', 'Good contrast + ratio > 0.5', c>60 and ratio_ok,
                             warn=(40 < c <= 60) or (0.3 <= contrast_ratio <= 0.5),
                             rec='Increase contrast for clearer details and avoid clipping.',
                             fix='Adjust contrast in your photo editor or retake with better lighting. Avoid overexposure.'))
        return out

    def exif_age(self) -> Dict[str,Any]:
        from datetime import datetime
        taken = exif_capture_datetime(self.img)
        if not taken:
            # No capture date in metadata. Do NOT fall back to the file's
            # modification time — that reflects when the file was saved/uploaded,
            # not when the photo was taken, so it would always look "recent".
            # Report honestly that recency could not be verified.
            return json_param('Photo Date', 'Unknown (no EXIF date)', 'Taken within last 6 months',
                              False, status='skipped',
                              rec='Capture date is missing from the photo, so recency could not be verified automatically.',
                              fix='DV photos must be taken within the last 6 months. Use an original camera/phone photo with the correct date/time set.')
        months = months_between(datetime.now(), taken)
        ok = months <= 6
        return json_param('Photo Date', taken.isoformat(sep=' ', timespec='seconds'), '≤ 6 months old', ok,
                          warn=not ok,
                          rec='Photo might be older than 6 months — please verify manually.',
                          fix='Take a new, recent photo. DV photos must be taken within the last 6 months.')

    def run(self) -> list[Dict[str,Any]]:
        results = [
            self.dimensions(),
            self.filesize(),
            self.compression_ratio(),
            self.fileformat(),
            self.color_model(),
            *self.brightness_contrast(),
            self.exif_age(),
        ]
        return results

