import io, math
import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageStat
from .utils import json_param


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
        self.img = pil_img.convert('RGB')

    # --- 1. Error Level Analysis ---
    def error_level_analysis(self):
        buf = io.BytesIO()
        self.img.save(buf, 'JPEG', quality=90)
        buf.seek(0)
        recompressed = Image.open(buf)
        diff = ImageChops.difference(self.img, recompressed)
        extrema = diff.getextrema()
        avg_diff = np.mean([sum(x)/3 for x in extrema])
        manipulated = avg_diff > 25
        return json_param(
            'Error Level Analysis',
            f'Δ={avg_diff:.2f}',
            'Low (uniform compression)',
            not manipulated,
            warn=10 < avg_diff <= 25,
            rec='Inconsistent compression detected — possible local editing.',
            fix='Use the original, unedited photo. Avoid any editing software that may introduce compression artifacts.'
        )

    # --- 2. Metadata consistency ---
    def metadata_consistency(self):
        exif = self.img.info.get('exif')
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

    # --- 3. Noise pattern uniformity ---
    def noise_uniformity(self):
        arr = np.asarray(self.img, dtype=np.float32)
        gray = arr.mean(axis=2)
        blocks = []
        h, w = gray.shape
        step = max(h//8, 1)
        for y in range(0, h, step):
            for x in range(0, w, step):
                block = gray[y:y+step, x:x+step]
                if block.size:
                    blocks.append(block.std())
        std_dev = np.std(blocks)
        ok = std_dev < 5
        return json_param(
            'Noise Uniformity',
            f'{std_dev:.2f}',
            'Consistent noise pattern',
            ok,
            warn=5 <= std_dev <= 10,
            rec='Uneven noise across image — may indicate splicing or editing.',
            fix='Use a single, original photo without any editing, filters, or compositing. Avoid combining multiple images.'
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

    def run(self):
        results = [
            self.error_level_analysis(),
            self.metadata_consistency(),
            self.noise_uniformity(),
            self.color_channel_consistency(),
        ]
        return results