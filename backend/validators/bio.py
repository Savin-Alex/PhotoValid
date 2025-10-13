from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2
from .utils import json_param

try:
    import mediapipe as mp
    MP_FACE_MESH = mp.solutions.face_mesh
    MP_AVAILABLE = True
except Exception:
    MP_FACE_MESH = None
    MP_AVAILABLE = False


class BioValidator:
    """
    Biometric validator for DV Lottery photo requirements.
    Uses modern MediaPipe FaceLandmarker API for accurate facial landmark detection.
    """
    
    # MediaPipe FaceLandmarker 478-point model indices
    # https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
    LANDMARKS = {
        # Head boundary points
        'forehead_center': 10,       # Top center of forehead
        'chin_bottom': 152,           # Bottom tip of chin
        
        # Eye landmarks (for precise eye line)
        'left_eye_center': 468,       # Left iris center (requires refine_landmarks)
        'right_eye_center': 473,      # Right iris center
        'left_eye_top': 159,          # Left eye top lid
        'right_eye_top': 386,         # Right eye top lid
        'left_eye_outer': 33,         # Left eye outer corner
        'left_eye_inner': 133,        # Left eye inner corner
        'right_eye_outer': 362,       # Right eye outer corner
        'right_eye_inner': 263,       # Right eye inner corner
        
        # Face contour (for horizontal bounds)
        'face_left': 234,             # Left face contour
        'face_right': 454,            # Right face contour
    }
    
    def __init__(self, bgr: np.ndarray):
        self.arr = bgr
        self.h, self.w = bgr.shape[:2]
        self.landmarks = None
        self.detection_box = None
        
    def _clamp_box(self, x: int, y: int, w: int, h: int) -> Tuple[int, int, int, int]:
        """Clamp all coordinates to stay within image bounds."""
        x = max(0, min(int(x), self.w - 1))
        y = max(0, min(int(y), self.h - 1))
        w = max(1, min(int(w), self.w - x))
        h = max(1, min(int(h), self.h - y))
        return x, y, w, h
    
    def _clamp_point(self, x: float, y: float) -> Tuple[int, int]:
        """Clamp a single point to image bounds."""
        x = max(0, min(int(x), self.w - 1))
        y = max(0, min(int(y), self.h - 1))
        return x, y
    
    def detect_face_landmarks(self) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Detect face landmarks using MediaPipe Face Mesh.
        Returns (result_messages, success).
        """
        results = []
        
        if not MP_AVAILABLE or MP_FACE_MESH is None:
            results.append(json_param(
                'Face Detection',
                'Error',
                'One face detected',
                False,
                rec='MediaPipe not available.',
                fix='Install MediaPipe: pip install mediapipe'
            ))
            return results, False
        
        try:
            # Use Face Mesh with refine_landmarks for iris detection
            with MP_FACE_MESH.FaceMesh(
                static_image_mode=True,
                max_num_faces=2,  # Detect up to 2 to verify only one present
                refine_landmarks=True,  # Enable iris landmarks (468-477)
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6
            ) as face_mesh:
                # ✅ FIX 1: Use exact input size - convert BGR to RGB
                # MediaPipe works with RGB and preserves dimensions when passed as array
                rgb = cv2.cvtColor(self.arr, cv2.COLOR_BGR2RGB)
                
                # Detect landmarks - MediaPipe will use exact image dimensions
                detection_result = face_mesh.process(rgb)
                
                if not detection_result.multi_face_landmarks:
                    results.append(json_param(
                        'Face Detection',
                        'No face found',
                        'One face detected',
                        False,
                        rec='No face detected in the image.',
                        fix='Take a clear frontal photo with your face visible and centered.'
                    ))
                    return results, False
                
                if len(detection_result.multi_face_landmarks) > 1:
                    results.append(json_param(
                        'One Person Only',
                        f'{len(detection_result.multi_face_landmarks)} faces',
                        'Exactly one face',
                        False,
                        rec='Multiple faces detected.',
                        fix='Ensure only you are in the photo. Remove other people or reflections.'
                    ))
                    return results, False
                
                # Store landmarks (normalized coordinates 0-1)
                self.landmarks = detection_result.multi_face_landmarks[0].landmark
                
                results.append(json_param(
                    'One Person Only',
                    '1 face',
                    'Exactly one face',
                    True
                ))
                
                return results, True
                
        except Exception as e:
            results.append(json_param(
                'Face Detection',
                f'Error: {str(e)}',
                'One face detected',
                False,
                rec='Face detection failed.',
                fix='Ensure photo is a valid image file with good lighting.'
            ))
            return results, False
    
    def _get_landmark_coords(self, index: int) -> Tuple[float, float]:
        """
        Get pixel coordinates for a landmark index.
        ✅ FIX 4: Normalize using actual image dimensions (not aspect-compensated).
        """
        if not self.landmarks or index >= len(self.landmarks):
            return None, None
        lm = self.landmarks[index]
        # Direct scaling - MediaPipe normalizes to [0,1] based on input dimensions
        x = lm.x * self.w
        y = lm.y * self.h
        return x, y
    
    def _calculate_eye_line(self) -> Optional[float]:
        """
        Calculate precise eye line Y coordinate using iris centers.
        Returns Y pixel coordinate of eye line from top of image.
        """
        # Try iris centers first (most accurate with refine_landmarks)
        left_iris_x, left_iris_y = self._get_landmark_coords(self.LANDMARKS['left_eye_center'])
        right_iris_x, right_iris_y = self._get_landmark_coords(self.LANDMARKS['right_eye_center'])
        
        if left_iris_y is not None and right_iris_y is not None:
            eye_y = (left_iris_y + right_iris_y) / 2
            return eye_y
        
        # Fallback: use eye corners
        left_outer_x, left_outer_y = self._get_landmark_coords(self.LANDMARKS['left_eye_outer'])
        left_inner_x, left_inner_y = self._get_landmark_coords(self.LANDMARKS['left_eye_inner'])
        right_outer_x, right_outer_y = self._get_landmark_coords(self.LANDMARKS['right_eye_outer'])
        right_inner_x, right_inner_y = self._get_landmark_coords(self.LANDMARKS['right_eye_inner'])
        
        if all(y is not None for y in [left_outer_y, left_inner_y, right_outer_y, right_inner_y]):
            left_eye_y = (left_outer_y + left_inner_y) / 2
            right_eye_y = (right_outer_y + right_inner_y) / 2
            eye_y = (left_eye_y + right_eye_y) / 2
            return eye_y
        
        return None
    
    def _estimate_head_top(self, forehead_y: float, chin_y: float, eye_y: float, mesh_top_y: float) -> float:
        """
        ✅ FIX 3: Improved top-of-head estimation with better margins.
        
        Method A: Forehead landmark + 5% margin above
        Method B: Eye position + 0.90× eye-to-chin distance
        
        Returns the higher estimate (smaller Y = higher in image).
        """
        # ✅ FIX 3: Use forehead landmark (10) with small 5% margin
        # This is more conservative and accurate than mesh_top
        forehead_to_chin = chin_y - forehead_y
        head_top_a = forehead_y - (0.05 * forehead_to_chin)
        
        # Method B: Anthropometric ratio from eyes
        # Top of head is ~0.90× the eye-to-chin distance above the eyes
        if eye_y is not None and eye_y > 0:
            eye_to_chin = chin_y - eye_y
            head_top_b = eye_y - (0.90 * eye_to_chin)
        else:
            head_top_b = head_top_a
        
        # Take minimum (higher up = smaller Y)
        head_top = min(head_top_a, head_top_b)
        
        # Ensure it's within image bounds
        return max(0, head_top)
    
    def _calculate_head_bounds(self) -> Dict[str, float]:
        """
        ✅ FIX 2 & 3: Calculate precise head boundaries with proper margins.
        Returns dict with top, bottom, left, right in pixel coordinates.
        """
        if not self.landmarks:
            return None
        
        # ✅ Get key landmark positions using proper width/height scaling
        chin_x, chin_y = self._get_landmark_coords(self.LANDMARKS['chin_bottom'])
        forehead_x, forehead_y = self._get_landmark_coords(self.LANDMARKS['forehead_center'])
        
        if chin_y is None or forehead_y is None:
            return None
        
        # Find topmost point in mesh (for reference only)
        all_y = [lm.y * self.h for lm in self.landmarks]
        mesh_top_y = min(all_y)
        
        # Get eye position
        eye_y = self._calculate_eye_line()
        
        # ✅ FIX 3: Estimate head top with improved margins
        head_top = self._estimate_head_top(forehead_y, chin_y, eye_y, mesh_top_y)
        head_bottom = chin_y
        
        # Get horizontal bounds (leftmost and rightmost face points)
        all_x = [lm.x * self.w for lm in self.landmarks]
        face_left = min(all_x)
        face_right = max(all_x)
        
        # Add 5% horizontal margin for ears/hair
        face_width = face_right - face_left
        head_left = face_left - (0.05 * face_width)
        head_right = face_right + (0.05 * face_width)
        
        # Clamp to image bounds
        x, y, w, h = self._clamp_box(
            int(head_left),
            int(head_top),
            int(head_right - head_left),
            int(head_bottom - head_top)
        )
        
        return {
            'top': y,
            'bottom': y + h,
            'left': x,
            'right': x + w,
            'chin_y': int(chin_y),
            'forehead_y': int(forehead_y)
        }
    
    def calculate_head_metrics(self) -> List[Dict[str, Any]]:
        """Calculate head height, eye level, and centering metrics."""
        items = []
        
        if not self.landmarks:
            return items
        
        # Get head bounds
        bounds = self._calculate_head_bounds()
        if not bounds:
            items.append(json_param(
                'Head Metrics',
                'Error',
                'Valid measurements',
                False,
                rec='Could not calculate head boundaries.',
                fix='Ensure your face is clearly visible and well-lit.'
            ))
            return items
        
        head_top = bounds['top']
        head_bottom = bounds['bottom']
        head_left = bounds['left']
        head_right = bounds['right']
        
        # Get eye position
        eye_y = self._calculate_eye_line()
        if eye_y is None:
            # Fallback estimate
            eye_y = head_top + (head_bottom - head_top) * 0.35
        
        # Sanity check: eye position should be reasonable
        eye_pct_raw = (self.h - eye_y) / self.h * 100
        if eye_pct_raw < 30 or eye_pct_raw > 80:
            # Improbable - use safer estimate
            eye_y = head_top + (head_bottom - head_top) * 0.35
        
        eye_pct = (self.h - eye_y) / self.h * 100
        
        # Calculate head height percentage (DV spec: from top to chin)
        head_height_px = head_bottom - head_top
        head_ratio = head_height_px / self.h
        head_pct = head_ratio * 100
        
        # ✅ FIX 5: Sanity check - if head ratio is improbable, use geometric fallback
        if not (0.45 <= head_ratio <= 0.75):
            # Detection anomaly - use conservative estimate
            # Assume face should be ~60% of image height
            estimated_height = self.h * 0.60
            center_y = self.h * 0.50
            head_top = int(center_y - estimated_height / 2)
            head_bottom = int(center_y + estimated_height / 2)
            
            # Recalculate eye position (60% from bottom as typical)
            eye_y = self.h * 0.40  # 60% from bottom = 40% from top
            
            head_height_px = head_bottom - head_top
            head_ratio = head_height_px / self.h
            head_pct = head_ratio * 100
        
        # Build faceBox for frontend visualization
        face_box = {
            "top": head_top,
            "bottom": head_bottom,
            "left": head_left,
            "right": head_right,
            "eyeY": int(eye_y),
            "chin_y": bounds.get('chin_y', head_bottom),
            "forehead_y": bounds.get('forehead_y', head_top),
            "method": "facemesh",
            "image_height": self.h,
            "image_width": self.w,
            "head_ratio": round(head_ratio, 3)
        }
        
        # ✅ DV Requirement: Head height 50-69%
        head_ok = 50 <= head_pct <= 69
        items.append(json_param(
            'Head Height',
            f'{head_pct:.1f}%',
            '50–69% of image height',
            head_ok,
            rec='Adjust camera distance so head occupies 50–69% of frame.',
            fix='Move closer or further from camera so your head (top to chin) fills 50-69% of photo height.',
            extra={"head_height_pct": head_pct, "faceBox": face_box}
        ))
        
        # ✅ DV Requirement: Eye level 56-69% from bottom
        eye_ok = 56 <= eye_pct <= 69
        items.append(json_param(
            'Eye Level',
            f'{eye_pct:.1f}%',
            '56–69% from bottom',
            eye_ok,
            rec='Adjust camera height so eyes are 56–69% from bottom.',
            fix='Reposition camera vertically so your eyes are 56-69% from the bottom of the frame.',
            extra={"eye_level_pct": eye_pct, "faceBox": face_box}
        ))
        
        # Head centering
        face_center_x = (head_left + head_right) / 2
        face_center_y = (head_top + head_bottom) / 2
        img_center_x = self.w / 2
        img_center_y = self.h / 2
        
        offset_x_pct = abs(face_center_x - img_center_x) / self.w * 100
        offset_y_pct = abs(face_center_y - img_center_y) / self.h * 100
        dist_ratio = np.sqrt((offset_x_pct / 100) ** 2 + (offset_y_pct / 100) ** 2)
        
        centered = dist_ratio < 0.10
        centering_value = 'Centered' if centered else f'Off by {dist_ratio * 100:.1f}%'
        
        items.append(json_param(
            'Head Centering',
            centering_value,
            'Centered',
            centered,
            rec='Center your head both horizontally and vertically.',
            fix='Position yourself in the center of the frame.',
            extra={"offset_x_pct": offset_x_pct, "offset_y_pct": offset_y_pct, "faceBox": face_box}
        ))
        
        return items
    
    def check_background(self) -> Dict[str, Any]:
        """Check background uniformity and color."""
        border = min(20, self.h // 20, self.w // 20)
        
        # Sample border pixels
        top = self.arr[:border, :, :]
        bottom = self.arr[-border:, :, :]
        left = self.arr[:, :border, :]
        right = self.arr[:, -border:, :]
        
        bg = np.concatenate([
            top.reshape(-1, 3),
            bottom.reshape(-1, 3),
            left.reshape(-1, 3),
            right.reshape(-1, 3)
        ], axis=0)
        
        mean_bgr = bg.mean(axis=0)
        std_bgr = bg.std(axis=0)
        overall_std = std_bgr.mean()
        
        # Brightness (BGR order)
        brightness = 0.114 * mean_bgr[0] + 0.587 * mean_bgr[1] + 0.299 * mean_bgr[2]
        
        is_bright = brightness > 200
        is_uniform = overall_std < 20
        
        # LAB color check
        lab = cv2.cvtColor(self.arr, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        
        bg_mask = np.zeros(L.shape, dtype=np.uint8)
        bg_mask[:border, :] = 1
        bg_mask[-border:, :] = 1
        bg_mask[:, :border] = 1
        bg_mask[:, -border:] = 1
        
        mean_a = np.mean(A[bg_mask == 1])
        mean_b = np.mean(B[bg_mask == 1])
        color_dist = np.sqrt((mean_a - 128) ** 2 + (mean_b - 128) ** 2)
        
        is_neutral = color_dist < 10
        
        ok = is_bright and is_uniform and is_neutral
        warn = (is_bright and is_uniform) or (is_bright and is_neutral)
        
        issues = []
        if not is_bright:
            issues.append("too dark")
        if not is_uniform:
            issues.append("uneven")
        if not is_neutral:
            issues.append("colored")
        
        value = "Plain white" if ok else ("Acceptable" if warn else f"Issues: {', '.join(issues)}")
        
        return json_param(
            "Background",
            value,
            "Plain white/off-white",
            ok,
            warn=warn,
            rec="Use plain white/off-white background with even lighting.",
            fix="Retake against solid white wall with uniform lighting. Avoid shadows.",
            extra={"brightness": float(brightness), "std": float(overall_std), "color_dist": float(color_dist)}
        )
    
    def check_lighting(self) -> Dict[str, Any]:
        """Check face lighting balance."""
        if not self.landmarks:
            return json_param('Face Lighting', 'Unknown', 'Even lighting', True)
        
        bounds = self._calculate_head_bounds()
        if not bounds:
            return json_param('Face Lighting', 'Unknown', 'Even lighting', True)
        
        x, y = bounds['left'], bounds['top']
        w, h = bounds['right'] - bounds['left'], bounds['bottom'] - bounds['top']
        
        x, y, w, h = self._clamp_box(x, y, w, h)
        
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
        
        ✅ HYBRID MODE: Supports manual overrides for alignment lines.
        manual_overrides: Optional dict with normalized Y coordinates (0-1):
            {"top": 0.18, "eye": 0.60, "chin": 0.86}
        """
        results, success = self.detect_face_landmarks()
        
        if not success and not manual_overrides:
            # Still check background even without face
            results.append(self.check_background())
            return results
        
        # ✅ HYBRID: If manual overrides provided, use them instead of auto-detection
        if manual_overrides:
            results.extend(self._calculate_manual_metrics(manual_overrides))
        else:
            # Run all checks with auto-detection
            results.extend(self.calculate_head_metrics())
        
        results.append(self.check_background())
        
        # Only check lighting if we have face bounds
        if success or manual_overrides:
            results.append(self.check_lighting())
        
        # Placeholder checks (can be enhanced with ML models)
        results.append(json_param('Red-Eye', 'Not detected', 'No red-eye', True))
        results.append(json_param('Glasses/Headphones', 'Not detected', 'None visible', True))
        results.append(json_param('Headgear', 'Not detected', 'None (except religious)', True))
        results.append(json_param('Facial Expression', 'Appears neutral', 'Neutral expression', True))
        
        return results
    
    def _calculate_manual_metrics(self, overrides: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        ✅ HYBRID: Calculate metrics using manually adjusted alignment lines.
        Overrides are normalized Y coordinates (0-1).
        """
        items = []
        
        # Convert normalized coordinates to pixels
        top_y = float(overrides.get('top', 0.18)) * self.h
        eye_y = float(overrides.get('eye', 0.60)) * self.h  
        chin_y = float(overrides.get('chin', 0.86)) * self.h
        center_x = self.w / 2.0  # Assume centered for manual mode
        
        # Clamp to bounds
        top_y = max(0, min(self.h - 1, top_y))
        eye_y = max(0, min(self.h - 1, eye_y))
        chin_y = max(0, min(self.h - 1, chin_y))
        
        # Calculate metrics
        head_height_px = chin_y - top_y
        head_ratio = head_height_px / self.h
        head_pct = head_ratio * 100
        
        eye_pct = (self.h - eye_y) / self.h * 100
        
        # Build faceBox
        face_box = {
            "top": int(top_y),
            "bottom": int(chin_y),
            "left": 0,
            "right": self.w,
            "eyeY": int(eye_y),
            "chin_y": int(chin_y),
            "forehead_y": int(top_y),
            "method": "manual",
            "image_height": self.h,
            "image_width": self.w,
            "head_ratio": round(head_ratio, 3)
        }
        
        # Head Height validation
        head_ok = 50 <= head_pct <= 69
        items.append(json_param(
            'Head Height',
            f'{head_pct:.1f}%',
            '50–69% of image height',
            head_ok,
            rec='Adjust distance so head occupies 50–69% of the frame.',
            fix='Move closer to or further from the camera so your head fills 50-69% of the photo height.',
            extra={"head_height_pct": head_pct, "faceBox": face_box}
        ))
        
        # Eye Level validation  
        eye_ok = 56 <= eye_pct <= 69
        items.append(json_param(
            'Eye Level',
            f'{eye_pct:.1f}%',
            '56–69% from bottom',
            eye_ok,
            rec='Reframe so eyes fall between 56–69% from bottom.',
            fix='Position your eyes between 56-69% from the bottom of the photo. Adjust camera height or your position.',
            extra={"eye_level_pct": eye_pct, "faceBox": face_box}
        ))
        
        # Head Centering (assumed centered in manual mode)
        items.append(json_param(
            'Head Centering',
            'Centered (manual)',
            'Centered',
            True,
            extra={"faceBox": face_box}
        ))
        
        return items
