import io, logging, math
import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageStat
from .utils import json_param

logger = logging.getLogger("photo_valid.tamper")


class TamperValidator:
    """
    Detect signs of digital manipulation (Photoshop / AI edits).
    Includes:
      1. Error Level Analysis (ELA)
      2. Metadata presence
      3. Noise pattern uniformity
      4. Color channel consistency
    """

    def __init__(self, pil_img: Image.Image):
        self.original_info = dict(getattr(pil_img, "info", {}) or {})
        self.img = pil_img.convert('RGB')

    # --- 1. Error Level Analysis ---
    def error_level_analysis(self):
        buf = io.BytesIO()
        self.img.save(buf, 'JPEG', quality=90)
        buf.seek(0)
        recompressed = Image.open(buf)
        diff = ImageChops.difference(self.img, recompressed)
        # getextrema() returns (min, max) per channel for the difference image.
        # Use the mean of the per-channel maxima as the peak error-level signal.
        # (The previous `sum(x)/3` averaged each (min, max) pair over 3, which was
        # not a meaningful quantity; thresholds below are scaled to this measure.)
        extrema = diff.getextrema()
        avg_diff = float(np.mean([hi for (_lo, hi) in extrema]))
        manipulated = avg_diff > 75
        return json_param(
            'Error Level Analysis',
            f'Δ={avg_diff:.2f}',
            'Low (uniform compression)',
            not manipulated,
            warn=30 < avg_diff <= 75,
            rec='Inconsistent compression detected — possible local editing.',
            fix='Use the original, unedited photo. Avoid any editing software that may introduce compression artifacts.'
        )

    # --- 2. Metadata consistency ---
    def metadata_consistency(self):
        exif = self.original_info.get('exif')
        ok = bool(exif)
        return json_param(
            'Metadata Consistency',
            'Present' if ok else 'Missing',
            'Consistent EXIF metadata',
            ok,
            warn=not ok,
            rec='EXIF data missing — edited images often strip metadata.',
            fix='Use the original photo from your camera/phone without any editing. Ensure camera date/time is set correctly.'
        )

    # --- 3. Image noise level (measured in flat regions) ---
    def image_noise(self):
        # Estimate the noise FLOOR: the variation within the flattest blocks
        # (background / smooth skin) approximates sensor/JPEG noise, not scene
        # detail. The old "noise uniformity" metric measured detail variation
        # across the whole frame, so any face-on-plain-background photo failed it.
        gray = np.asarray(self.img.convert("L"), dtype=np.float32)
        h, w = gray.shape
        grid = 16
        bh, bw = max(h // grid, 1), max(w // grid, 1)
        stds = []
        for y in range(0, h - bh + 1, bh):
            for x in range(0, w - bw + 1, bw):
                stds.append(float(gray[y:y + bh, x:x + bw].std()))
        flat = float(np.percentile(stds, 10)) if stds else 0.0
        ok = flat < 6.0
        return json_param(
            'Image Noise',
            f'{flat:.1f}',
            'Low noise (clean image)',
            ok,
            warn=6.0 <= flat <= 12.0,
            rec='Image looks grainy/noisy.',
            fix='Use better lighting and a lower ISO setting; avoid heavy noise/sharpening filters.'
        )

    # --- 4. Color channel consistency ---
    def color_channel_consistency(self):
        arr = np.asarray(self.img, dtype=np.float32)
        means = [arr[:,:,i].mean() for i in range(3)]
        diffs = np.std(means)
        ok = diffs < 12
        return json_param(
            'Color Channel Balance',
            f'Δ={diffs:.2f}',
            'Balanced RGB channels',
            ok,
            warn=12 <= diffs <= 20,
            rec='Unnatural channel balance — verify photo not color-edited.',
            fix='Use natural lighting and avoid color filters or adjustments. Take photo with balanced, natural colors.'
        )

    def _safe_check(self, fn, name):
        try:
            return fn()
        except Exception:
            logger.exception("%s check failed", name)
            return json_param(
                name,
                "Skipped",
                "Analyzer completed",
                False,
                status="skipped",
                rec=f"{name} could not run; details are in the server logs.",
                fix="Try a valid JPEG from the original camera/phone and redeploy if this persists.",
            )

    def run(self):
        checks = [
            (self.error_level_analysis, "Error Level Analysis"),
            (self.metadata_consistency, "Metadata Consistency"),
            (self.image_noise, "Image Noise"),
            (self.color_channel_consistency, "Color Channel Balance"),
        ]
        return [self._safe_check(fn, name) for fn, name in checks]