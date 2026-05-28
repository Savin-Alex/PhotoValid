from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import logging
import math
import threading
import numpy as np
from .utils import json_param

try:
    import cv2
    CV2_AVAILABLE = True
except Exception as exc:
    cv2 = None
    CV2_AVAILABLE = False
    CV2_IMPORT_ERROR = str(exc)
else:
    CV2_IMPORT_ERROR = None

try:
    import mediapipe as mp
    MP_FACE_MESH = mp.solutions.face_mesh
    MP_FACE_DETECTION = mp.solutions.face_detection
    MP_AVAILABLE = CV2_AVAILABLE
except Exception:
    MP_FACE_MESH = None
    MP_FACE_DETECTION = None
    MP_AVAILABLE = False

# ✅ Debug mode: Set to True to enable visual debug markers
DEBUG = False

logger = logging.getLogger("photo_valid.bio")

# Creating MediaPipe graphs is expensive; rebuilding them on every request adds
# large latency. Cache one instance of each model and reuse it across requests.
# MediaPipe graph objects are NOT thread-safe, so every .process() call is
# serialized with a lock. (The validate endpoint is otherwise effectively
# serialized on a single free-tier worker, so the lock is essentially free.)
_MP_LOCK = threading.Lock()
_FACE_DETECTION = None
_FACE_MESH = None


def _get_face_detection():
    """Lazily build the cached FaceDetection graph. Returns None if construction
    fails (e.g. no GL context on a headless host) so callers can degrade cleanly."""
    global _FACE_DETECTION
    if _FACE_DETECTION is None and MP_AVAILABLE and MP_FACE_DETECTION is not None:
        try:
            _FACE_DETECTION = MP_FACE_DETECTION.FaceDetection(
                model_selection=0,  # Short range for portraits
                min_detection_confidence=0.5,
            )
        except Exception:
            logger.exception("Failed to initialize MediaPipe FaceDetection")
            _FACE_DETECTION = None
    return _FACE_DETECTION


def _get_face_mesh():
    """Lazily build the cached FaceMesh graph. Returns None on init failure."""
    global _FACE_MESH
    if _FACE_MESH is None and MP_AVAILABLE and MP_FACE_MESH is not None:
        try:
            _FACE_MESH = MP_FACE_MESH.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,  # Enable iris detection
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception:
            logger.exception("Failed to initialize MediaPipe FaceMesh")
            _FACE_MESH = None
    return _FACE_MESH


class BioValidator:
    """
    ✅ PRODUCTION-GRADE Biometric Validator for DV Lottery Photos
    
    Features:
    - Hybrid "Contrast + Landmark" top-of-head detection
    - Always returns faceBox (even on failure) for overlay visualization
    - Improved eye-line fallback & smoothing with sanity checks
    - Safe coordinate clamping throughout
    - Optional debug mode for visual verification
    - Manual override support for hybrid mode
    
    Uses MediaPipe Face Detection + Face Mesh for accurate measurements.
    """
    
    # MediaPipe landmark indices (468-point face mesh)
    LANDMARK_INDICES = {
        'forehead_top': 10,      # Top center of forehead
        'chin_bottom': 152,      # Bottom of chin
        'left_eye_outer': 33,    # Left eye outer corner
        'right_eye_outer': 263,  # Right eye outer corner
        'left_iris': 468,        # Left iris center (requires refine_landmarks)
        'right_iris': 473,       # Right iris center (requires refine_landmarks)
    }
    
    def __init__(self, bgr: np.ndarray):
        self.arr = bgr
        self.h, self.w = bgr.shape[:2]
        self.face_count = 0
        self.landmarks = None  # populated by calculate() in auto mode
        self.detection_failed = False  # True if the model could not init/process
        self.debug_image = bgr.copy() if DEBUG else None
    
    def detect_face_box(self):
        """Return bounding box [xmin, ymin, width, height] or None."""
        if not MP_AVAILABLE or MP_FACE_DETECTION is None or not CV2_AVAILABLE or cv2 is None:
            self.detection_failed = True
            return None

        fd = _get_face_detection()
        if fd is None:
            self.detection_failed = True
            return None

        try:
            rgb = cv2.cvtColor(self.arr, cv2.COLOR_BGR2RGB)
            with _MP_LOCK:
                res = fd.process(rgb)
        except Exception:
            logger.exception("MediaPipe FaceDetection failed to process the image")
            self.detection_failed = True
            return None

        if not res.detections:
            self.face_count = 0
            return None

        self.face_count = len(res.detections)
        bb = res.detections[0].location_data.relative_bounding_box

        # Convert to absolute pixels
        xmin = int(bb.xmin * self.w)
        ymin = int(bb.ymin * self.h)
        w = int(bb.width * self.w)
        h = int(bb.height * self.h)

        # Clamp to image bounds
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        if xmin + w > self.w:
            w = self.w - xmin
        if ymin + h > self.h:
            h = self.h - ymin

        return (xmin, ymin, w, h)
    
    def detect_landmarks(self):
        """Return face mesh landmarks or None."""
        if not MP_AVAILABLE or MP_FACE_MESH is None or not CV2_AVAILABLE or cv2 is None:
            self.detection_failed = True
            return None

        fm = _get_face_mesh()
        if fm is None:
            self.detection_failed = True
            return None

        try:
            rgb = cv2.cvtColor(self.arr, cv2.COLOR_BGR2RGB)
            with _MP_LOCK:
                res = fm.process(rgb)
        except Exception:
            logger.exception("MediaPipe FaceMesh failed to process the image")
            self.detection_failed = True
            return None

        if not res.multi_face_landmarks:
            return None

        return res.multi_face_landmarks[0]
    
    def _find_true_head_top(self, lms, gray: np.ndarray) -> Tuple[int, str]:
        """
        ✅ HYBRID "Contrast + Landmark" Top-of-Head Detection
        
        Starting just above the forehead landmark, scan upward in a narrow column
        to detect a strong brightness boundary (head → background).
        
        Args:
            lms: MediaPipe face mesh landmarks
            gray: Grayscale image array
        
        Returns:
            Tuple of (y_coordinate, method_used)
            - y_coordinate: Absolute Y pixel (0-based) of true head top
            - method_used: "contrast_edge" | "mesh_margin" | "fallback"
        """
        # Get forehead landmark position
        lmpts = [(lm.x * self.w, lm.y * self.h) for lm in lms.landmark]
        fore_x, fore_y = lmpts[self.LANDMARK_INDICES['forehead_top']]
        chin_x, chin_y = lmpts[self.LANDMARK_INDICES['chin_bottom']]
        
        fx = int(fore_x)
        fy = int(fore_y)
        
        # ✅ IMPROVEMENT: Calculate search range based on face height
        face_height = chin_y - fore_y
        max_up = min(int(face_height * 0.5), fy)  # Don't search more than 50% of face height up
        
        # ✅ IMPROVEMENT: Wider column for more robust edge detection (±4px instead of ±2px)
        x1 = max(0, fx - 4)
        x2 = min(gray.shape[1], fx + 4)
        
        # Extract column from (forehead - max_up) to forehead
        y_start = max(0, fy - max_up)
        column = gray[y_start:fy, x1:x2]
        
        if column.size == 0:
            # Fallback to 25% margin above forehead
            fallback_top = max(0, int(fore_y - 0.25 * face_height))
            return fallback_top, "fallback"
        
        # Calculate brightness profile (average across the stripe width)
        profile = np.mean(column, axis=1)
        
        # Find gradient (brightness changes)
        grad = np.abs(np.diff(profile))
        
        # ✅ IMPROVEMENT: Adaptive threshold based on image contrast
        # Use 8 for high-contrast images, lower for subtle transitions
        threshold = 8
        edges = np.where(grad > threshold)[0]
        
        if len(edges) > 0:
            # ✅ Take the FIRST strong edge (topmost, closest to image top)
            edge_idx = edges[0]
            boundary_y = y_start + edge_idx
            
            # ✅ SANITY CHECK: Make sure detected edge is reasonable
            # Should be above forehead and not ridiculously far up
            if boundary_y < fy and (fy - boundary_y) < (0.5 * face_height):
                if DEBUG and self.debug_image is not None:
                    cv2.circle(self.debug_image, (fx, boundary_y), 3, (0, 255, 0), -1)  # Green dot
                return int(boundary_y), "contrast_edge"
        
        # ✅ No clear edge found - use adaptive margin based on face proportions
        # 10% margin is conservative and works for most hairstyles
        mesh_top = max(0, int(fore_y - 0.10 * face_height))
        
        if DEBUG and self.debug_image is not None:
            cv2.circle(self.debug_image, (fx, mesh_top), 3, (255, 255, 0), -1)  # Cyan dot
        
        return mesh_top, "mesh_margin"
    
    def calculate(self, manual_overrides: Optional[Dict[str, float]] = None):
        """
        ✅ PRODUCTION-GRADE Head Measurement Calculation
        
        Returns dict with faceBox, head_ratio, eye_level, center_offset.
        Supports manual overrides (normalized Y coordinates 0-1).
        Always returns valid faceBox for frontend overlay.
        """
        # ✅ MANUAL MODE: If manual overrides provided, use them.
        # NOTE: manual mode does NOT detect a face — face_count stays 0 so we never
        # imply "one face detected" from manually placed guide lines.
        if manual_overrides:
            top_y = float(manual_overrides.get('top', 0.18)) * self.h
            eye_y = float(manual_overrides.get('eye', 0.60)) * self.h
            chin_y = float(manual_overrides.get('chin', 0.86)) * self.h
            
            # Clamp to image bounds
            top_y = max(0, min(self.h - 1, top_y))
            eye_y = max(0, min(self.h - 1, eye_y))
            chin_y = max(0, min(self.h - 1, chin_y))
            
            head_ratio = (chin_y - top_y) / self.h * 100.0
            eye_level = (self.h - eye_y) / self.h * 100.0
            
            faceBox = {
                "top": int(top_y),
                "bottom": int(chin_y),
                "eyeY": int(eye_y),
                "left": 0,
                "right": self.w,
                "centerX": self.w / 2,
                "image_height": self.h,
                "image_width": self.w,
                "method": "manual"
            }
            
            return {
                "faceBox": faceBox,
                "head_ratio": head_ratio,
                "eye_level": eye_level,
                "center_offset": 0.0
            }
        
        # ✅ AUTO DETECTION
        face = self.detect_face_box()
        lms = self.detect_landmarks()
        self.landmarks = lms  # cache for downstream heuristics (e.g. glasses)

        # ✅ SAFE FALLBACK: If no face detected, return sensible defaults
        if face is None:
            fallback_box = {
                "top": int(0.20 * self.h),
                "bottom": int(0.86 * self.h),
                "eyeY": int(0.60 * self.h),
                "left": int(0.10 * self.w),
                "right": int(0.90 * self.w),
                "centerX": self.w / 2,
                "image_height": self.h,
                "image_width": self.w,
                "method": "fallback"
            }
            return {
                "faceBox": fallback_box,
                "head_ratio": None,
                "eye_level": None,
                "center_offset": None
            }
        
        xmin, ymin, fw, fh = face
        
        # ✅ LANDMARK-BASED PRECISION: If landmarks exist, compute precise measurements
        if lms is not None:
            lmpts = [(lm.x * self.w, lm.y * self.h) for lm in lms.landmark]
            
            # Key landmarks
            chin_x, chin_y = lmpts[self.LANDMARK_INDICES['chin_bottom']]
            fore_x, fore_y = lmpts[self.LANDMARK_INDICES['forehead_top']]
            
            # ✅ IMPROVED EYE LINE: Try iris centers first, fall back to eye corners
            try:
                # Iris centers (most accurate for eye level)
                iris_l_x, iris_l_y = lmpts[self.LANDMARK_INDICES['left_iris']]
                iris_r_x, iris_r_y = lmpts[self.LANDMARK_INDICES['right_iris']]
                eye_y = (iris_l_y + iris_r_y) / 2.0
                eye_method = "iris"
            except (IndexError, KeyError):
                # Fallback to eye corners
                ex1, ey1 = lmpts[self.LANDMARK_INDICES['left_eye_outer']]
                ex2, ey2 = lmpts[self.LANDMARK_INDICES['right_eye_outer']]
                eye_y = (ey1 + ey2) / 2.0
                eye_method = "corners"
            
            # ✅ SANITY CHECK: If eye_y is wildly wrong (< 30% or > 80%), use box heuristic
            eye_pct_check = (self.h - eye_y) / self.h * 100.0
            if eye_pct_check < 30 or eye_pct_check > 80:
                eye_y = ymin + 0.5 * fh
                eye_method = "box_fallback"
            
            # ✅ HYBRID CONTRAST + LANDMARK: Find true top of head
            gray = cv2.cvtColor(self.arr, cv2.COLOR_BGR2GRAY)
            top_y, top_method = self._find_true_head_top(lms, gray)
            
            # ✅ SANITY CHECK: Head ratio should be 45-75% (reasonable range)
            head_ratio_check = (chin_y - top_y) / self.h * 100.0
            if head_ratio_check < 45 or head_ratio_check > 75:
                # Fallback to geometric estimate
                face_height = chin_y - fore_y
                top_y = max(0, int(fore_y - 0.15 * face_height))
                top_method = "geometric_fallback"
            
            # Eye centers for horizontal centering
            try:
                center_x = (lmpts[self.LANDMARK_INDICES['left_eye_outer']][0] + 
                           lmpts[self.LANDMARK_INDICES['right_eye_outer']][0]) / 2.0
            except (IndexError, KeyError):
                center_x = fore_x
            
            if DEBUG and self.debug_image is not None:
                # Draw debug markers
                cv2.circle(self.debug_image, (int(chin_x), int(chin_y)), 4, (255, 0, 0), -1)  # Blue: chin
                cv2.circle(self.debug_image, (int(fore_x), int(fore_y)), 4, (0, 165, 255), -1)  # Orange: forehead
                cv2.circle(self.debug_image, (int(center_x), int(eye_y)), 4, (0, 255, 255), -1)  # Yellow: eyes
        else:
            # ✅ BOX-BASED FALLBACK: Use bounding box proportions
            top_y = ymin
            chin_y = ymin + fh
            eye_y = ymin + 0.5 * fh
            center_x = xmin + fw / 2.0
            top_method = "box"
            eye_method = "box"
        
        # ✅ COMPUTE FINAL METRICS
        head_ratio = (chin_y - top_y) / self.h * 100.0
        eye_level = (self.h - eye_y) / self.h * 100.0
        center_offset = abs(center_x - self.w / 2.0) / self.w * 100.0
        
        # ✅ ALWAYS RETURN FACEBOX with all coordinates
        faceBox = {
            "top": int(top_y),
            "bottom": int(chin_y),
            "eyeY": int(eye_y),
            "left": xmin,
            "right": xmin + fw,
            "centerX": center_x,
            "image_height": self.h,
            "image_width": self.w,
            "method": "facemesh" if lms is not None else "facedetection",
            "top_method": top_method if lms is not None else "box",
            "eye_method": eye_method if lms is not None else "box"
        }
        
        return {
            "faceBox": faceBox,
            "head_ratio": head_ratio,
            "eye_level": eye_level,
            "center_offset": center_offset
        }
    
    def check_background(self) -> Dict[str, Any]:
        """Check the background using the top corners (in LAB color space).

        In a correctly framed head-and-shoulders portrait the head touches the top
        and the shoulders/clothing fill the bottom, so a full-border sample is
        dominated by the subject (hair, ears, clothing) — not the background. The
        top-left and top-right corners are the most reliable pure-background area.
        """
        if not CV2_AVAILABLE or cv2 is None:
            return json_param(
                "Background",
                "Skipped",
                "OpenCV available",
                False,
                status="skipped",
                rec=f"OpenCV could not be imported: {CV2_IMPORT_ERROR}",
                fix="Deploy with opencv-python-headless installed in the active Render Python environment.",
            )

        bw = max(int(self.w * 0.16), 8)
        bh = max(int(self.h * 0.16), 8)
        corners = [self.arr[:bh, :bw], self.arr[:bh, self.w - bw:]]
        lab = np.concatenate(
            [cv2.cvtColor(c, cv2.COLOR_BGR2LAB).reshape(-1, 3) for c in corners], axis=0
        ).astype(np.float32)

        mean_L = float(lab[:, 0].mean())
        std_L = float(lab[:, 0].std())
        mean_A = float(lab[:, 1].mean())
        mean_B = float(lab[:, 2].mean())
        delta_E = float(np.sqrt((mean_A - 128) ** 2 + (mean_B - 128) ** 2))

        # LAB L is 0-255 (OpenCV 8-bit). White ~255; off-white/light-gray still high.
        is_bright = mean_L > 200
        is_uniform = std_L < 25
        is_neutral = delta_E < 10

        ok = is_bright and is_uniform and is_neutral
        warn = (not ok) and (mean_L > 180) and (std_L < 40) and (delta_E < 15)

        value = f"L={mean_L:.0f} ΔE={delta_E:.1f} σ={std_L:.1f}"

        return json_param(
            "Background",
            value,
            "Plain white/off-white",
            ok,
            warn=warn,
            rec="Use a plain white/off-white background with even lighting.",
            fix="Retake against a solid white wall. Avoid shadows, patterns, and props.",
            extra={"mean_L": mean_L, "std_L": std_L, "delta_E": delta_E}
        )
    
    def check_sharpness(self, faceBox) -> Dict[str, Any]:
        """Measure sharpness using Laplacian variance on face ROI."""
        if not CV2_AVAILABLE or cv2 is None:
            return json_param(
                'Sharpness',
                'Skipped',
                'OpenCV available',
                False,
                status='skipped',
                rec=f'OpenCV could not be imported: {CV2_IMPORT_ERROR}',
                fix='Deploy with opencv-python-headless installed in the active Render Python environment.'
            )

        if faceBox is None:
            return json_param('Sharpness', 'Unknown', 'Sharp focus', True)
        
        x = max(0, faceBox["left"])
        y = max(0, faceBox["top"])
        w = min(self.w - x, faceBox["right"] - faceBox["left"])
        h = min(self.h - y, faceBox["bottom"] - faceBox["top"])
        
        if w <= 0 or h <= 0:
            return json_param('Sharpness', 'Error', 'Sharp focus', True)
        
        gray = cv2.cvtColor(self.arr, cv2.COLOR_BGR2GRAY)
        roi = gray[y:y + h, x:x + w]
        
        if roi.size == 0:
            return json_param('Sharpness', 'Error', 'Sharp focus', True)
        
        laplacian_var = float(cv2.Laplacian(roi, cv2.CV_64F).var())
        
        ok = laplacian_var >= 80
        warn = 50 <= laplacian_var < 80
        
        return json_param(
            'Sharpness',
            f'{laplacian_var:.1f} variance',
            '≥80 (sharp focus)',
            ok,
            warn=warn,
            rec='Ensure sharp focus on face.',
            fix='Retake with proper focus, steady camera, and good lighting.',
            extra={"laplacian_variance": float(laplacian_var)}
        )
    
    def check_lighting(self, faceBox) -> Dict[str, Any]:
        """Check face lighting balance."""
        if not CV2_AVAILABLE or cv2 is None:
            return json_param(
                'Face Lighting',
                'Skipped',
                'OpenCV available',
                False,
                status='skipped',
                rec=f'OpenCV could not be imported: {CV2_IMPORT_ERROR}',
                fix='Deploy with opencv-python-headless installed in the active Render Python environment.'
            )

        if faceBox is None:
            return json_param('Face Lighting', 'Unknown', 'Even lighting', True)
        
        x = max(0, faceBox["left"])
        y = max(0, faceBox["top"])
        w = min(self.w - x, faceBox["right"] - faceBox["left"])
        h = min(self.h - y, faceBox["bottom"] - faceBox["top"])
        
        if w <= 0 or h <= 0:
            return json_param('Face Lighting', 'Unknown', 'Even lighting', True)
        
        face = self.arr[y:y + h, x:x + w]
        if face.size == 0:
            return json_param('Face Lighting', 'Unknown', 'Even lighting', True)
        
        mid = w // 2
        left_half = face[:, :mid]
        right_half = face[:, mid:]
        
        left_bright = left_half.mean()
        right_bright = right_half.mean()
        
        ratio = min(left_bright, right_bright) / max(left_bright, right_bright) if max(left_bright, right_bright) > 0 else 0
        
        ok = ratio >= 0.85
        warn = 0.75 <= ratio < 0.85
        
        value = "Even" if ok else (f"Slightly uneven ({ratio * 100:.0f}%)" if warn else f"Uneven ({ratio * 100:.0f}%)")
        
        return json_param(
            'Face Lighting',
            value,
            'Even lighting',
            ok,
            warn=warn,
            rec='Use balanced lighting on both sides.',
            fix='Use soft, diffused front lighting. Avoid side lighting.',
            extra={"balance_ratio": float(ratio)}
        )
    
    def check_glasses(self) -> Dict[str, Any]:
        """Best-effort eyeglasses heuristic.

        Eyeglasses are PROHIBITED in DV photos (since 2016-11-01). A glasses
        bridge crosses the nose bridge between the eyes as a strong edge, whereas
        a bare nose bridge is smooth. We measure edge density in that small
        region and only WARN on a positive — we never confidently "clear" a photo
        of glasses (a negative stays "manual review"), so a false negative cannot
        green-light a prohibited photo.

        NOTE: the threshold below is a conservative heuristic and has NOT been
        calibrated on labelled glasses/no-glasses photos. Treat it as an aid, not
        a definitive check.
        """
        name = 'Glasses/Headphones'
        expected = 'No eyeglasses (prohibited)'
        skip_rec = ('Eyeglasses are NOT allowed in DV photos. Auto-detection is '
                    'best-effort — please verify manually.')
        skip_fix = 'Remove eyeglasses (and any headphones) and retake the photo.'

        lms = getattr(self, 'landmarks', None)
        if not CV2_AVAILABLE or cv2 is None or lms is None:
            return json_param(name, 'Not auto-checked', expected, False, status='skipped',
                              rec=skip_rec, fix=skip_fix)

        try:
            pts = lms.landmark
            lx = int(pts[133].x * self.w)   # left inner eye corner
            rx = int(pts[362].x * self.w)   # right inner eye corner
            by = int(pts[168].y * self.h)   # nose bridge between the eyes

            x1, x2 = sorted((lx, rx))
            span = max(x2 - x1, 8)
            x1 = max(0, x1 - int(span * 0.15))
            x2 = min(self.w, x2 + int(span * 0.15))
            band = max(int(0.05 * self.h), 6)
            y1 = max(0, by - band)
            y2 = min(self.h, by + band)

            if x2 - x1 < 4 or y2 - y1 < 4:
                return json_param(name, 'Not auto-checked', expected, False, status='skipped',
                                  rec=skip_rec, fix=skip_fix)

            gray = cv2.cvtColor(self.arr, cv2.COLOR_BGR2GRAY)
            roi = gray[y1:y2, x1:x2]
            edges = cv2.Canny(roi, 100, 200)
            density = float(np.count_nonzero(edges)) / float(edges.size or 1)
        except Exception:
            return json_param(name, 'Not auto-checked', expected, False, status='skipped',
                              rec=skip_rec, fix=skip_fix)

        if density >= 0.06:
            return json_param(
                name, f'Possible eyeglasses (edge density {density * 100:.0f}%)', expected,
                False, warn=True,
                rec='Possible eyeglasses detected — eyeglasses are NOT allowed in DV photos. '
                    'Verify and retake without glasses if present.',
                fix='Remove eyeglasses and retake the photo.',
                extra={"bridge_edge_density": density})

        return json_param(name, 'None auto-detected', expected, False, status='skipped',
                          rec=skip_rec, fix=skip_fix)

    def _pts(self):
        """Pixel-space landmark points from the cached Face Mesh, or None."""
        lms = getattr(self, 'landmarks', None)
        if lms is None:
            return None
        return [(lm.x * self.w, lm.y * self.h) for lm in lms.landmark]

    @staticmethod
    def _dist(a, b) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def check_eyes_open(self) -> Dict[str, Any]:
        """Both-eyes-open check via Eye Aspect Ratio (EAR).

        EAR = (vertical eyelid gaps) / (2 * horizontal eye width). Open eyes give
        ~0.25-0.35; a closed eye approaches 0 (Soukupova & Cech, 2016). Uses the
        standard 6-point MediaPipe eye landmark sets.
        """
        name, expected = 'Eyes Open', 'Both eyes open'
        skip = lambda: json_param(name, 'Not auto-checked', expected, False, status='skipped',
                                  rec='Could not auto-check; verify both eyes are open.',
                                  fix='Retake with both eyes fully open, looking at the camera.')
        p = self._pts()
        if p is None:
            return skip()
        try:
            def ear(o, t1, t2, i, b2, b1):
                horiz = self._dist(p[o], p[i])
                if horiz <= 0:
                    return None
                return (self._dist(p[t1], p[b1]) + self._dist(p[t2], p[b2])) / (2.0 * horiz)
            ear_r = ear(33, 160, 158, 133, 153, 144)
            ear_l = ear(263, 385, 387, 362, 373, 380)
        except (IndexError, TypeError):
            return skip()
        if ear_r is None or ear_l is None:
            return skip()
        ear = min(ear_r, ear_l)
        ok = ear >= 0.20
        warn = 0.15 <= ear < 0.20
        return json_param(name, f'EAR {ear:.2f}', expected, ok, warn=warn,
                          rec='One or both eyes appear closed or squinting.',
                          fix='Retake with both eyes fully open (no squinting).',
                          extra={"ear": ear})

    def check_gaze(self) -> Dict[str, Any]:
        """Looking-at-camera check via horizontal iris centering.

        For each eye, ratio = (iris_x - outer_x) / (inner_x - outer_x); ~0.5 means
        the iris sits centered between the eye corners (gaze toward camera).
        Requires refined iris landmarks (468/473).
        """
        name, expected = 'Gaze (Looking at Camera)', 'Looking at camera'
        skip = lambda: json_param(name, 'Not auto-checked', expected, False, status='skipped',
                                  rec='Could not auto-check; verify you are looking straight at the camera.',
                                  fix='Look directly at the camera lens.')
        p = self._pts()
        if p is None or len(p) <= 473:
            return skip()
        try:
            def ratio(outer, inner, iris):
                denom = p[inner][0] - p[outer][0]
                if denom == 0:
                    return None
                return (p[iris][0] - p[outer][0]) / denom
            r_r = ratio(33, 133, 468)
            r_l = ratio(263, 362, 473)
        except (IndexError, TypeError):
            return skip()
        if r_r is None or r_l is None:
            return skip()
        off = (abs(r_r - 0.5) + abs(r_l - 0.5)) / 2.0
        ok = off <= 0.18
        warn = 0.18 < off <= 0.30
        return json_param(name, f'{off * 100:.0f}% off-center', expected, ok, warn=warn,
                          rec='Eyes appear to be looking away from the camera.',
                          fix='Look directly at the camera lens.',
                          extra={"gaze_offset": off})

    def check_expression(self) -> Dict[str, Any]:
        """Neutral-expression check via mouth openness (Mouth Aspect Ratio).

        MAR = inner-lip vertical gap / mouth width. A closed neutral mouth is
        small (~0.0-0.15); an open mouth / broad grin raises it.
        """
        name, expected = 'Facial Expression', 'Neutral, mouth closed'
        skip = lambda: json_param(name, 'Not auto-checked', expected, False, status='skipped',
                                  rec='Could not auto-check expression; verify it is neutral with mouth closed.',
                                  fix='Use a neutral expression with your mouth closed.')
        p = self._pts()
        if p is None:
            return skip()
        try:
            width = self._dist(p[61], p[291])
            if width <= 0:
                return skip()
            mar = self._dist(p[13], p[14]) / width
        except (IndexError, TypeError, ZeroDivisionError):
            return skip()
        ok = mar <= 0.30
        warn = 0.30 < mar <= 0.50
        return json_param(name, f'mouth openness {mar:.2f}', expected, ok, warn=warn,
                          rec='Mouth appears open or smiling — expression should be neutral.',
                          fix='Close your mouth and use a neutral expression.',
                          extra={"mouth_aspect_ratio": mar})

    def check_redeye(self) -> Dict[str, Any]:
        """Best-effort red-eye heuristic.

        Looks for strongly red pixels inside the iris regions (flash red-eye makes
        the pupil glow red). Flag-only: warns on a likely positive and never
        confidently clears (a negative stays manual review). Uncalibrated.
        """
        name, expected = 'Red-Eye', 'No red-eye'
        skip = lambda: json_param(name, 'Not auto-checked', expected, False, status='skipped',
                                  rec='Red-eye could not be auto-checked — please verify manually.',
                                  fix='Ensure there is no red-eye reflection; retake without direct flash if needed.')
        p = self._pts()
        if not CV2_AVAILABLE or cv2 is None or p is None or len(p) <= 473:
            return skip()
        try:
            worst = 0.0
            r = max(3, int(0.015 * self.w))
            for iris_idx in (468, 473):
                cx, cy = int(p[iris_idx][0]), int(p[iris_idx][1])
                y1, y2 = max(0, cy - r), min(self.h, cy + r)
                x1, x2 = max(0, cx - r), min(self.w, cx + r)
                roi = self.arr[y1:y2, x1:x2]  # BGR
                if roi.size == 0:
                    continue
                B = roi[:, :, 0].astype(np.float32)
                G = roi[:, :, 1].astype(np.float32)
                R = roi[:, :, 2].astype(np.float32)
                redish = (R > 120) & (R > G * 1.6) & (R > B * 1.6)
                frac = float(np.count_nonzero(redish)) / float(redish.size or 1)
                worst = max(worst, frac)
        except Exception:
            return skip()

        if worst >= 0.20:
            return json_param(
                name, f'Possible red-eye ({worst * 100:.0f}% red in iris)', expected,
                False, warn=True,
                rec='Possible red-eye detected — eyes must not show a red flash reflection.',
                fix='Retake without direct flash, or correct the red-eye, then re-upload.',
                extra={"iris_red_fraction": worst})
        return skip()

    def check_head_tilt(self) -> Dict[str, Any]:
        """In-plane head tilt (roll) from the angle of the inter-eye line.

        A level head puts both eyes on a horizontal line (~0 degrees). Uses iris
        centers when available, otherwise the outer eye corners.
        """
        name, expected = 'Head Tilt', 'Upright (not tilted)'
        skip = lambda: json_param(name, 'Not auto-checked', expected, False, status='skipped',
                                  rec='Could not auto-check; keep your head upright and level.',
                                  fix='Hold your head straight and level (not tilted left or right).')
        p = self._pts()
        if p is None:
            return skip()
        try:
            if len(p) > 473:
                ax, ay = p[468]    # right iris center
                bx, by = p[473]    # left iris center
            else:
                ax, ay = p[33]     # right eye outer corner
                bx, by = p[263]    # left eye outer corner
            angle = abs(math.degrees(math.atan2(by - ay, bx - ax)))
            if angle > 90:
                angle = 180 - angle
        except (IndexError, TypeError):
            return skip()
        ok = angle <= 5.0
        warn = 5.0 < angle <= 10.0
        return json_param(name, f'{angle:.1f}°', expected, ok, warn=warn,
                          rec='Head appears tilted; keep it upright and level.',
                          fix='Hold your head straight and level (not tilted left or right).',
                          extra={"tilt_degrees": angle})

    def _headgear_skip(self) -> Dict[str, Any]:
        return json_param(
            'Headgear', 'Not auto-checked', 'None (except daily religious)', False, status='skipped',
            rec='Hats/head coverings are not auto-detected — please verify manually.',
            fix='Remove any hat or head covering unless worn daily for religious reasons.')

    def _skipped_biometrics(self, reason: str) -> List[Dict[str, Any]]:
        """Named 'skipped' results for the required biometric checks when face
        analysis cannot run (model unavailable or init/process failure).

        Critical checks reported as skipped force a non-pass overall status in
        main._summarize_response — they must never look like a clean pass.
        """
        fix = ("Use a clear, face-forward photo, and ensure the server has MediaPipe + "
               "opencv-python-headless on Python 3.11.")
        required = [
            ("Face Detection", "One face detected"),
            ("One Person Only", "Exactly one face"),
            ("Head Height", "50–69% of image height"),
            ("Eye Level", "56–69% from bottom"),
            ("Head Centering", "Centered ±5%"),
            ("Sharpness", "≥80 (sharp focus)"),
            ("Face Lighting", "Even lighting"),
        ]
        results = [
            json_param(name, "Not checked", expected, False, status="skipped", rec=reason, fix=fix)
            for name, expected in required
        ]
        # Background is corner-based and does not need a detected face.
        results.append(self.check_background())
        # The remaining checks self-skip when landmarks are unavailable.
        results.append(self.check_redeye())
        results.append(self.check_glasses())
        results.append(self._headgear_skip())
        results.append(self.check_eyes_open())
        results.append(self.check_gaze())
        results.append(self.check_expression())
        results.append(self.check_head_tilt())
        return results

    def _manual_results(self, manual_overrides: Dict[str, float]) -> List[Dict[str, Any]]:
        """Manual mode: the user dragged the top/eye/chin guide lines. We measure
        head height and eye level from those lines, but no face was detected — so
        Face Detection and One Person Only are reported as unverified (skipped),
        never as a pass, and centering can't be derived from horizontal lines.
        """
        calc = self.calculate(manual_overrides)
        fb = calc.get("faceBox")
        hr = calc.get("head_ratio")
        el = calc.get("eye_level")
        note = " (measured from your manually placed lines; face presence was not auto-verified)"

        results = [
            json_param("Face Detection", "Manual mode", "One face detected", False, status="skipped",
                       rec="Face was not auto-detected in manual mode.",
                       fix="Run automatic validation (clear the manual lines) to verify a face is present."),
            json_param("One Person Only", "Manual mode", "Exactly one face", False, status="skipped",
                       rec="Number of people was not auto-verified in manual mode.",
                       fix="Ensure only the applicant is visible; run automatic validation to confirm."),
        ]
        if hr is not None:
            ok = 50 <= hr <= 69
            warn = (45 <= hr < 50) or (69 < hr <= 72)
            results.append(json_param("Head Height", f"{hr:.1f}% (manual)", "50–69% of image height", ok, warn=warn,
                                      rec="Top of head to chin must cover 50–69% of image." + note,
                                      fix="Drag the top and chin lines to your true head extent, or reframe.",
                                      extra={"head_height_pct": hr, "faceBox": fb, "manual": True}))
        if el is not None:
            ok = 56 <= el <= 69
            warn = (53 <= el < 56) or (69 < el <= 72)
            results.append(json_param("Eye Level", f"{el:.1f}% (manual)", "56–69% from bottom", ok, warn=warn,
                                      rec="Eyes should sit 56–69% from the bottom." + note,
                                      fix="Drag the eye line to your eyes, or reframe.",
                                      extra={"eye_level_pct": el, "faceBox": fb, "manual": True}))
        results.append(json_param("Head Centering", "Manual mode", "Centered ±5%", False, status="skipped",
                                  rec="Centering can't be measured from the horizontal manual lines.",
                                  fix="Run automatic validation to measure horizontal centering."))
        results.append(self.check_background())
        results.append(self.check_sharpness(fb))
        results.append(self.check_lighting(fb))
        # Heuristics need landmarks (unavailable in manual mode) -> self-skip.
        results.append(self.check_redeye())
        results.append(self.check_glasses())
        results.append(self._headgear_skip())
        results.append(self.check_eyes_open())
        results.append(self.check_gaze())
        results.append(self.check_expression())
        results.append(self.check_head_tilt())
        return results

    def run(self, manual_overrides: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """Run all biometric validations (auto), or measure manually placed lines.

        On model unavailability/failure the required checks are emitted as named
        'skipped' results (never silently dropped, never a generic skip).
        """
        if manual_overrides:
            return self._manual_results(manual_overrides)

        if not MP_AVAILABLE:
            return self._skipped_biometrics(
                "Face analysis is unavailable in this runtime (MediaPipe/OpenCV not installed).")

        try:
            calc = self.calculate(None)
        except Exception:
            logger.exception("Biometric analysis raised")
            self.detection_failed = True
            calc = None

        if self.detection_failed or calc is None:
            return self._skipped_biometrics(
                "Face analysis could not run in this environment; please verify the photo manually.")

        fb = calc.get("faceBox")
        hr = calc.get("head_ratio")
        el = calc.get("eye_level")
        co = calc.get("center_offset")

        results = []

        # Genuine no-face: the model ran but found nothing -> FAIL (critical).
        if fb is None or fb.get("method") == "fallback":
            results.append(json_param(
                "Face Detection", "Not found", "One face detected", False,
                rec="No face detected; retake clearly face-forward.",
                fix="Ensure the face is visible, well-lit, and facing the camera."))
            results.append(json_param(
                "One Person Only", "Not found", "Exactly one face", False,
                rec="No face detected, so one-person could not be confirmed.",
                fix="Retake with exactly one person, facing the camera."))
            results.append(self.check_background())
            return results

        # Face detected -> always emit an explicit Face Detection result.
        results.append(json_param(
            "Face Detection",
            "1 face" if self.face_count <= 1 else f"{self.face_count} faces",
            "One face detected", self.face_count >= 1,
            rec="No face detected; retake clearly face-forward.",
            fix="Ensure the face is visible, well-lit, and facing the camera."))

        # One person check
        one_person = self.face_count == 1
        results.append(json_param(
            'One Person Only',
            f'{self.face_count or 1} face' if self.face_count <= 1 else f'{self.face_count} faces',
            'Exactly one face',
            one_person,
            rec='Only one person may appear in the photo.',
            fix='Retake the photo with only the applicant visible.'
        ))
        
        # Head Height check (50-69%)
        if hr is not None:
            ok = 50 <= hr <= 69
            warn = (45 <= hr < 50) or (69 < hr <= 72)
            results.append(json_param(
                "Head Height",
                f"{hr:.1f}%",
                "50–69% of image height",
                ok,
                warn=warn,
                rec="Top of head to chin must cover 50–69% of image.",
                fix="Adjust camera distance or reframe to fill 50-69% with your head.",
                extra={"head_height_pct": hr, "faceBox": fb}
            ))
        
        # Eye Level check (56-69% from bottom)
        if el is not None:
            ok = 56 <= el <= 69
            warn = (53 <= el < 56) or (69 < el <= 72)
            results.append(json_param(
                "Eye Level",
                f"{el:.1f}%",
                "56–69% from bottom",
                ok,
                warn=warn,
                rec="Eyes should appear slightly above midline.",
                fix="Raise or lower camera so eyes fall in 56–69% zone.",
                extra={"eye_level_pct": el, "faceBox": fb}
            ))
        
        # Head Centering check (±5%)
        if co is not None:
            ok = co <= 5.0
            warn = 5.0 < co <= 8.0
            results.append(json_param(
                "Head Centering",
                f"{co:.1f}% offset",
                "Centered ±5%",
                ok,
                warn=warn,
                rec="Align your face to center of image.",
                fix="Shift slightly left/right to center face.",
                extra={"offset_pct": co, "faceBox": fb}
            ))
        
        # Background, sharpness, lighting
        results.append(self.check_background())
        results.append(self.check_sharpness(fb))
        results.append(self.check_lighting(fb))
        
        # Checks that are NOT yet auto-detected. Report them honestly as
        # "manual review required" (status=skipped) instead of silently passing —
        # eyeglasses in particular are a top DV disqualifier (prohibited since 2016-11-01).
        results.append(self.check_redeye())
        results.append(self.check_glasses())
        results.append(self._headgear_skip())
        results.append(self.check_eyes_open())
        results.append(self.check_gaze())
        results.append(self.check_expression())
        results.append(self.check_head_tilt())

        return results
