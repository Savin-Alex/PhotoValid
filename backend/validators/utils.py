from __future__ import annotations
import io, math, datetime as dt
from typing import Dict, Any, Tuple
from PIL import Image, ImageOps
import numpy as np
import piexif

RGB = Tuple[int, int, int]

def load_image_bytes(b: bytes) -> Image.Image:
    """Load image from bytes and fix EXIF orientation."""
    img = Image.open(io.BytesIO(b))
    original_format = img.format
    
    # ✅ FIX A: Correct EXIF orientation before detection
    # This ensures MediaPipe sees an upright face regardless of how the photo was taken
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass  # No EXIF orientation data or error - continue with original
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.format = original_format
    return img

def pil_to_cv(img: Image.Image) -> np.ndarray:
    # Pillow RGB -> OpenCV BGR
    return np.asarray(img.convert("RGB"))[:, :, ::-1].copy()

def is_grayscale(arr: np.ndarray) -> bool:
    if arr.ndim == 3 and arr.shape[2] == 3:
        r, g, b = arr[:,:,2], arr[:,:,1], arr[:,:,0]
        return np.allclose(r, g) and np.allclose(r, b)
    return True

def brightness(arr: np.ndarray) -> float:
    # luma-like measure 0..100
    if arr.ndim == 3:
        b,g,r = arr[:,:,0].astype(np.float32), arr[:,:,1].astype(np.float32), arr[:,:,2].astype(np.float32)
        y = 0.114*b + 0.587*g + 0.299*r
    else:
        y = arr.astype(np.float32)
    return float(np.clip(y.mean()/255*100, 0, 100))

def contrast(arr: np.ndarray) -> float:
    if arr.ndim == 3:
        y = arr.mean(axis=2)
    else:
        y = arr
    c = (y.max()-y.min())/255*100
    return float(c)

def sharpness_laplacian(arr: np.ndarray) -> float:
    import cv2
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if arr.ndim==3 else arr
    val = cv2.Laplacian(gray, cv2.CV_64F).var()
    # map to 0..100 roughly
    return float(np.clip(val/10.0, 0, 100))

def exif_capture_datetime(img: Image.Image) -> dt.datetime | None:
    try:
        exif = piexif.load(img.info.get('exif', b''))
        dt_str = exif.get('0th', {}).get(piexif.ImageIFD.DateTime)
        if not dt_str:
            dt_str = exif.get('Exif', {}).get(piexif.ExifIFD.DateTimeOriginal)
        if isinstance(dt_str, bytes):
            dt_str = dt_str.decode('utf-8', 'ignore')
        if dt_str:
            # format 'YYYY:MM:DD HH:MM:SS'
            return dt.datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
    except Exception:
        return None
    return None

def months_between(a: dt.datetime, b: dt.datetime) -> float:
    return abs((a - b).days) / 30.4375

def center_distance_ratio(w: int, h: int, box: Tuple[int,int,int,int]) -> float:
    # box = (x, y, w, h)
    cx, cy = w/2, h/2
    bx, by, bw, bh = box
    bcx, bcy = bx + bw/2, by + bh/2
    dist = math.hypot(bcx-cx, bcy-cy)
    maxd = math.hypot(w/2, h/2)
    return float(dist/maxd)

def _bool_scalar(value: Any) -> bool:
    """Convert Python/NumPy scalar-like values to bool without ambiguous arrays."""
    if isinstance(value, np.ndarray):
        return bool(np.all(value)) if value.size else False
    if isinstance(value, np.generic):
        return bool(value.item())
    return bool(value)

def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value

def json_param(name: str, value: Any=None, expected: str|None=None,
               ok: bool|None=None, warn: bool=False, rec: str|None=None,
               fix: str|None=None, extra: Dict[str, Any]|None=None,
               status: str|None=None):
    check_name = str(name or "Unknown Check")
    if status:
        normalized_status = status if status in {"pass", "warning", "fail", "skipped"} else "fail"
    elif _bool_scalar(ok):
        normalized_status = 'pass'
    elif _bool_scalar(warn):
        normalized_status = 'warning'
    else:
        normalized_status = 'fail'

    d = {
        "name": check_name,
        "parameter": check_name,
        "value": _json_safe(value),
        "expected": _json_safe(expected),
        "status": normalized_status,
        "ok": normalized_status == "pass",
    }
    if rec and normalized_status != 'pass':
        d["recommendation"] = rec
    if fix and normalized_status != 'pass':
        d["how_to_fix"] = fix
    if extra:
        d.update(_json_safe(extra))
    return d

