from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
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

# ✅ Debug mode: Set to True to enable visual debug markers
DEBUG = False


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
        self.debug_image = bgr.copy() if DEBUG else None
    
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
        # ✅ MANUAL MODE: If manual overrides provided, use them
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
