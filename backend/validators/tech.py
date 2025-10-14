from __future__ import annotations
import math
from typing import Dict, Any
from PIL import Image
import numpy as np
from .utils import is_grayscale, brightness, contrast, sharpness_laplacian, exif_capture_datetime, months_between, json_param

MAX_KB = 240

class TechValidator:
    def __init__(self, pil_img: Image.Image, raw_bytes: bytes, content_type: str):
        self.img = pil_img
        self.raw = raw_bytes
        self.content_type = content_type
        self.arr = np.asarray(pil_img.convert('RGB'))[:,:,::-1]  # BGR for cv

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

    def fileformat(self) -> Dict[str, Any]:
        ok = self.content_type in ('image/jpeg','image/jpg')
        return json_param('File Format', self.content_type, 'JPEG', ok, 
                         rec='Save as JPEG (.jpg)',
                         fix='Convert your photo to JPEG format. Use "Save As" and select JPEG (.jpg) in your photo editor.')

    def color_model(self) -> Dict[str, Any]:
        ok = not is_grayscale(self.arr)
        return json_param('Color Model', 'sRGB' if ok else 'Grayscale', 'sRGB (color)', ok, 
                         rec='Use a color photo (no grayscale).',
                         fix='Take a new color photo or convert your black-and-white photo to color using photo editing software.')

    def brightness_contrast(self) -> list[Dict[str,Any]]:
        b = brightness(self.arr)
        c = contrast(self.arr)
        out = []
        
        # ✅ FIXED: Adjusted brightness range to 45-65% (more natural exposure)
        out.append(json_param('Brightness', f'{b:.0f}%', '45–65%', 45 <= b <= 65, warn=35<=b<=75, 
                             rec='Adjust lighting; avoid over/underexposure.',
                             fix='Retake photo with better lighting. Use natural light or ensure flash is properly adjusted.'))
        
        # Enhanced contrast with ratio test
        gray = self.arr.mean(axis=2) if len(self.arr.shape) == 3 else self.arr
        min_val, max_val = gray.min(), gray.max()
        contrast_ratio = (max_val - min_val) / max_val if max_val > 0 else 0
        
        # ✅ NEW: Contrast ratio test (max-min)/max > 0.5 for acceptable contrast
        ratio_ok = contrast_ratio > 0.5
        out.append(json_param('Contrast', f'{c:.1f}% (ratio: {contrast_ratio:.2f})', 'Good contrast + ratio > 0.5', c>60 and ratio_ok, 
                             warn=40<c<=60 or (0.3<=contrast_ratio<=0.5), 
                             rec='Increase contrast for clearer details and avoid clipping.',
                             fix='Adjust contrast in photo editor or retake with better lighting to enhance detail visibility. Avoid overexposure.'))
        return out

    def sharpness(self) -> Dict[str,Any]:
        # Simplified sharpness check without OpenCV (Python 3.13 compatible)
        try:
            import cv2
            # If OpenCV is available, use it
            gray = cv2.cvtColor(self.arr, cv2.COLOR_BGR2GRAY)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            brightness = gray.mean() + 1
            sharp_norm = lap_var / brightness
            edges = cv2.Canny(gray, 50, 150)
        except ImportError:
            # Fallback: basic sharpness estimation using PIL
            from PIL import Image, ImageFilter
            pil_img = Image.fromarray(self.arr)
            gray = pil_img.convert('L')
            
            # Simple edge detection using PIL's built-in filters
            edges = gray.filter(ImageFilter.FIND_EDGES)
            edge_array = np.array(edges)
            sharp_norm = np.var(edge_array) / (np.mean(edge_array) + 1)
        edge_density = np.count_nonzero(edges) / edges.size * 100
        score = 0.6 * min(sharp_norm / 10, 100) + 0.4 * edge_density
        
        # ✅ FIXED: Adjust threshold - warn at 40+ instead of fail
        ok = score > 60
        warn = 40 <= score <= 60  # Changed from 40 < score <= 60 to include 40
        return json_param('Sharpness', f'{score:.1f} (normalized)', '≥ 60 (sharp)', ok, warn=warn,
                          rec='Ensure focus; avoid motion blur or low light.',
                          fix='Retake photo with proper focus and steady camera. Avoid zoom or resizing.')

    def exif_age(self) -> Dict[str,Any]:
        import os
        from datetime import datetime
        taken = exif_capture_datetime(self.img)
        if not taken:
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(self.raw)
                    tmp.flush()
                    mod_time = datetime.fromtimestamp(os.path.getmtime(tmp.name))
                os.unlink(tmp.name)
                taken = mod_time
            except Exception:
                taken = None
        if not taken:
            return json_param('Photo Date', 'Unknown', 'Taken within last 6 months', True, warn=True,
                              rec='EXIF date missing — please ensure your photo was taken within the last 6 months.',
                              fix='Take a new photo within the last 6 months. Ensure your camera/phone has correct date/time settings.')
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
            self.fileformat(),
            self.color_model(),
            *self.brightness_contrast(),
            self.sharpness(),
            self.exif_age(),
        ]
        return results

