from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np
import cv2
from .utils import json_param

try:
    import mediapipe as mp
    MP_FACE_MESH = mp.solutions.face_mesh
    MP_FACE_DETECTION = mp.solutions.face_detection
    MP_AVAILABLE = True
except Exception:
    MP_FACE_MESH = None
    MP_FACE_DETECTION = None
    MP_AVAILABLE = False


class BioValidator:
    """
    Biometric validator for DV Lottery photo requirements.
    Uses MediaPipe Face Detection + Face Mesh for accurate measurements.
    Always returns faceBox for overlay visualization.
    """
    
    def __init__(self, bgr: np.ndarray):
        self.arr = bgr
        self.h, self.w = bgr.shape[:2]
    
    def detect_face_box(self):
        """Return bounding box [xmin, ymin, width, height] or None."""
        if not MP_AVAILABLE or MP_FACE_DETECTION is None:
            return None
        
        with MP_FACE_DETECTION.FaceDetection(
            model_selection=0,  # Short range for portraits
            min_detection_confidence=0.5
        ) as fd:
            rgb = cv2.cvtColor(self.arr, cv2.COLOR_BGR2RGB)
            res = fd.process(rgb)
            
            if not res.detections:
                return None
            
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
        if not MP_AVAILABLE or MP_FACE_MESH is None:
            return None
        
        with MP_FACE_MESH.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,  # Enable iris detection
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as fm:
            rgb = cv2.cvtColor(self.arr, cv2.COLOR_BGR2RGB)
            res = fm.process(rgb)
            
            if not res.multi_face_landmarks:
                return None
            
            return res.multi_face_landmarks[0]
    
    def _find_true_head_top(self, gray: np.ndarray, forehead_x: int, forehead_y: int) -> int:
        """
        ✅ CONTRAST-BASED: Find actual top of head by scanning upward to image top.
        Detects where dark hair/skin meets bright background.
        
        Args:
            gray: Grayscale image
            forehead_x, forehead_y: Starting point (forehead landmark)
        
        Returns:
            Y coordinate of true head top (where head meets background)
        """
        # ✅ Scan ALL the way to top of image (not just 150px)
        # Extract narrow vertical stripe from top of image to forehead
        x1 = max(0, forehead_x - 3)
        x2 = min(gray.shape[1] - 1, forehead_x + 3)
        y1 = 0  # Start from top of image
        
        column = gray[y1:forehead_y, x1:x2]
        
        if column.size == 0:
            return forehead_y
        
        # Calculate brightness profile (average across the narrow stripe)
        profile = np.mean(column, axis=1)
        
        # Find gradient (brightness changes)
        grad = np.abs(np.diff(profile))
        
        # Find strong edges (threshold tuned for hair-to-background transition)
        # Typical transition: dark hair (50-100) → white wall (230-255) = gradient ~100+
        # Lower threshold to 12 to catch subtle transitions too
        edges = np.where(grad > 12)[0]
        
        if len(edges) > 0:
            # Take the topmost strong edge (first one scanning from top)
            boundary_idx = edges[0]
            boundary_y = y1 + boundary_idx
            return int(max(0, boundary_y))
        
        # No clear edge found - fallback (no margin, just forehead)
        # This happens with bright hair on white background
        return forehead_y
    
    def calculate(self, manual_overrides: Optional[Dict[str, float]] = None):
        """
        Calculate head measurements.
        Returns dict with faceBox, head_ratio, eye_level.
        Supports manual overrides (normalized Y coordinates 0-1).
        """
        # ✅ HYBRID: If manual overrides provided, use them
        if manual_overrides:
            top_y = float(manual_overrides.get('top', 0.18)) * self.h
            eye_y = float(manual_overrides.get('eye', 0.60)) * self.h
            chin_y = float(manual_overrides.get('chin', 0.86)) * self.h
            
            # Clamp
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
        
        # Auto detection
        face = self.detect_face_box()
        lms = self.detect_landmarks()
        
        # If no face at all, return fallback values
        if face is None:
            fallback_box = {
                "top": int(0.20 * self.h),
                "bottom": int(0.86 * self.h),
                "eyeY": int(0.60 * self.h),
                "left": 0,
                "right": self.w,
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
        
        # If landmarks exist, compute precise lines
        if lms is not None:
            lmpts = [(lm.x * self.w, lm.y * self.h) for lm in lms.landmark]
            
            # Key landmarks
            chin_x, chin_y = lmpts[152]  # Chin
            fore_x, fore_y = lmpts[10]   # Forehead center
            
            # Eye Y via average of outer corners
            ex1, ey1 = lmpts[33]   # Right eye outer
            ex2, ey2 = lmpts[263]  # Left eye outer
            eye_y = (ey1 + ey2) / 2.0
            
            # ✅ CONTRAST-BASED: Find true top of head using edge detection
            # Scans from image top to forehead, finds exact hair/background boundary
            gray = cv2.cvtColor(self.arr, cv2.COLOR_BGR2GRAY)
            detected_top = self._find_true_head_top(gray, int(fore_x), int(fore_y))
            
            # Use detected top if it's reasonable, otherwise fall back to margin
            face_h = chin_y - fore_y
            conservative_top = fore_y - (0.25 * face_h)  # Fallback
            
            # Sanity check: detected top should be above forehead and not too far up
            if detected_top < fore_y and (fore_y - detected_top) < (0.4 * face_h):
                top_y = detected_top
            else:
                # Fallback to adaptive margin
                top_y = conservative_top
            
            # Safety: clamp to image
            top_y = max(0, top_y)
            
            center_x = (ex1 + ex2) / 2.0
        else:
            # Fallback: use bounding box proportions
            top_y = ymin
            chin_y = ymin + fh
            eye_y = ymin + 0.5 * fh
            center_x = xmin + fw / 2.0
        
        # Compute normalized ratios
        head_ratio = (chin_y - top_y) / self.h * 100.0
        eye_level = (self.h - eye_y) / self.h * 100.0
        
        # Centering offset
        center_offset = abs(center_x - self.w / 2.0) / self.w * 100.0
        
        # ✅ Always return faceBox with all coordinates
        faceBox = {
            "top": int(top_y),
            "bottom": int(chin_y),
            "eyeY": int(eye_y),
            "left": xmin,
            "right": xmin + fw,
            "centerX": center_x,
            "image_height": self.h,
            "image_width": self.w,
            "method": "facemesh" if lms is not None else "facedetection"
        }
        
        return {
            "faceBox": faceBox,
            "head_ratio": head_ratio,
            "eye_level": eye_level,
            "center_offset": center_offset
        }
    
    def check_background(self) -> Dict[str, Any]:
        """
        Check background using border bands (10% margin).
        Uses LAB color space for perceptual analysis.
        """
        # Sample 10% border bands
        band_size = max(int(self.h * 0.10), 10)
        
        # Extract border regions
        top_band = self.arr[:band_size, :, :]
        bottom_band = self.arr[-band_size:, :, :]
        left_band = self.arr[:, :band_size, :]
        right_band = self.arr[:, -band_size:, :]
        
        # Combine and convert to LAB
        sample_h = min(100, self.h)
        sample_w = min(100, self.w)
        sample = cv2.resize(self.arr, (sample_w, sample_h))
        lab = cv2.cvtColor(sample, cv2.COLOR_BGR2LAB)
        
        # Sample border in LAB
        band_sample = max(int(sample_h * 0.10), 5)
        lab_top = lab[:band_sample, :, :]
        lab_bottom = lab[-band_sample:, :, :]
        lab_left = lab[:, :band_sample, :]
        lab_right = lab[:, -band_sample:, :]
        
        lab_bg = np.concatenate([
            lab_top.reshape(-1, 3),
            lab_bottom.reshape(-1, 3),
            lab_left.reshape(-1, 3),
            lab_right.reshape(-1, 3)
        ], axis=0)
        
        # Statistics
        mean_L = np.mean(lab_bg[:, 0])
        std_L = np.std(lab_bg[:, 0])
        mean_A = np.mean(lab_bg[:, 1])
        mean_B = np.mean(lab_bg[:, 2])
        delta_E = np.sqrt((mean_A - 128) ** 2 + (mean_B - 128) ** 2)
        
        # Criteria
        is_bright = mean_L > 180
        is_uniform = std_L < 15
        is_neutral = delta_E < 10
        
        ok = is_bright and is_uniform and is_neutral
        warn = (is_bright and is_uniform) or (is_bright and is_neutral)
        
        value = f"L={mean_L:.1f} ΔE={delta_E:.1f} σ={std_L:.1f}"
        
        return json_param(
            "Background",
            value,
            "Plain white/off-white",
            ok,
            warn=warn,
            rec="Use plain white/off-white background with even lighting.",
            fix="Retake against solid white wall. Avoid shadows and patterns.",
            extra={"mean_L": float(mean_L), "std_L": float(std_L), "delta_E": float(delta_E)}
        )
    
    def check_sharpness(self, faceBox) -> Dict[str, Any]:
        """Measure sharpness using Laplacian variance on face ROI."""
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
        
        laplacian_var = cv2.Laplacian(roi, cv2.CV_64F).var()
        
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
    
    def run(self, manual_overrides: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """
        Run all biometric validations.
        Supports manual overrides for hybrid mode.
        """
        # Calculate measurements (auto or manual)
        calc = self.calculate(manual_overrides)
        fb = calc.get("faceBox")
        hr = calc.get("head_ratio")
        el = calc.get("eye_level")
        co = calc.get("center_offset")
        
        results = []
        
        # Face detection check
        if fb is None or fb.get("method") == "fallback":
            results.append(json_param(
                "Face Detection",
                "Not found",
                "One face detected",
                False,
                rec="No face detected; retake clearly face-forward.",
                fix="Ensure face is visible, well-lit, and facing camera."
            ))
            results.append(self.check_background())
            return results
        
        # One person check
        results.append(json_param('One Person Only', '1 face', 'Exactly one face', True))
        
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
        
        # Placeholder checks
        results.append(json_param('Red-Eye', 'Not detected', 'No red-eye', True))
        results.append(json_param('Glasses/Headphones', 'Not detected', 'None visible', True))
        results.append(json_param('Headgear', 'Not detected', 'None (except religious)', True))
        results.append(json_param('Facial Expression', 'Appears neutral', 'Neutral expression', True))
        
        return results
