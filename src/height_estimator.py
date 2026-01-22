"""
Automatic camera height estimation from aerial images.
Uses object detection and heuristics to estimate the camera/drone altitude.
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional


class CameraHeightEstimator:
    """
    Estimates camera height from aerial imagery using visual cues.
    
    Reference objects and their typical sizes:
    - Cars: ~4.5m long, ~1.8m wide
    - Trucks/Excavators: ~8-12m long
    - Shipping containers: 6m (20ft) or 12m (40ft) long
    - People: ~0.5m wide from above
    - Dumpsters: ~2-3m wide
    """
    
    # Reference object sizes (meters)
    REFERENCE_SIZES = {
        'car': {'length': 4.5, 'width': 1.8},
        'truck': {'length': 10.0, 'width': 2.5},
        'excavator': {'length': 8.0, 'width': 3.0},
        'container': {'length': 6.0, 'width': 2.4},  # 20ft container
        'dumpster': {'length': 3.0, 'width': 2.0},
        'person': {'length': 0.5, 'width': 0.5},
    }
    
    def __init__(self):
        self.detected_objects = []
    
    def estimate(
        self,
        image: np.ndarray,
        fov_degrees: float = 70.0,
    ) -> Dict:
        """
        Estimate camera height from image.
        
        Args:
            image: BGR image
            fov_degrees: Camera field of view (typical drone: 70-90°)
        
        Returns:
            Dict with estimated height and confidence
        """
        h, w = image.shape[:2]
        
        estimates = []
        
        # Method 1: Detect vehicle-like objects (yellow/orange for construction equipment)
        vehicle_est = self._estimate_from_vehicles(image, fov_degrees)
        if vehicle_est:
            estimates.append(vehicle_est)
        
        # Method 2: Detect rectangular objects (containers, dumpsters)
        rect_est = self._estimate_from_rectangles(image, fov_degrees)
        if rect_est:
            estimates.append(rect_est)
        
        # Method 3: Image resolution/detail heuristic
        detail_est = self._estimate_from_detail_level(image)
        estimates.append(detail_est)
        
        # Method 4: Scene analysis (how much is visible)
        scene_est = self._estimate_from_scene_coverage(image)
        estimates.append(scene_est)
        
        # Combine estimates (weighted average based on confidence)
        if estimates:
            total_weight = sum(e['confidence'] for e in estimates)
            if total_weight > 0:
                weighted_height = sum(e['height'] * e['confidence'] for e in estimates) / total_weight
            else:
                weighted_height = 30.0  # Default
            
            # Clamp to reasonable range
            final_height = max(3.0, min(150.0, weighted_height))
            
            return {
                'estimated_height_m': round(final_height, 1),
                'confidence': min(1.0, total_weight / len(estimates)),
                'method_estimates': estimates,
                'recommendation': self._get_height_category(final_height),
            }
        
        # Fallback
        return {
            'estimated_height_m': 30.0,
            'confidence': 0.2,
            'method_estimates': [],
            'recommendation': 'medium_drone',
        }
    
    def _estimate_from_vehicles(self, image: np.ndarray, fov: float) -> Optional[Dict]:
        """Detect construction vehicles (yellow/orange) and estimate height."""
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Yellow/orange construction equipment
        yellow_mask = cv2.inRange(hsv, (15, 100, 100), (35, 255, 255))
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        vehicle_sizes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:  # Too small
                continue
            
            rect = cv2.minAreaRect(contour)
            box_w, box_h = rect[1]
            if box_w < 10 or box_h < 10:
                continue
            
            # Vehicle-like aspect ratio (1:2 to 1:4)
            aspect = max(box_w, box_h) / (min(box_w, box_h) + 1)
            if 1.5 < aspect < 5:
                vehicle_sizes.append(max(box_w, box_h))
        
        if vehicle_sizes:
            # Assume largest yellow object is excavator (~8m)
            largest_px = max(vehicle_sizes)
            reference_m = 8.0  # Excavator length
            
            # Calculate height: h = (reference_m * image_width) / (2 * object_px * tan(fov/2))
            fov_rad = np.radians(fov)
            height = (reference_m * w) / (2 * largest_px * np.tan(fov_rad / 2))
            
            return {
                'method': 'vehicle_detection',
                'height': height,
                'confidence': 0.7,
                'details': f'Detected {len(vehicle_sizes)} yellow objects, largest {largest_px:.0f}px'
            }
        
        return None
    
    def _estimate_from_rectangles(self, image: np.ndarray, fov: float) -> Optional[Dict]:
        """Detect rectangular objects (containers, dumpsters)."""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangles = []
        for contour in contours:
            if cv2.contourArea(contour) < 1000:
                continue
            
            # Approximate to polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # Look for 4-sided shapes (rectangles)
            if len(approx) == 4:
                rect = cv2.minAreaRect(contour)
                box_w, box_h = rect[1]
                if box_w > 20 and box_h > 20:
                    aspect = max(box_w, box_h) / (min(box_w, box_h) + 1)
                    # Container-like (2:1 to 3:1) or dumpster-like (1.5:1)
                    if 1.3 < aspect < 3.5:
                        rectangles.append((max(box_w, box_h), aspect))
        
        if rectangles:
            # Assume largest rectangle is a container or dumpster
            largest_px, aspect = max(rectangles, key=lambda x: x[0])
            
            # Guess reference size based on aspect ratio
            if aspect > 2.2:
                reference_m = 6.0  # 20ft container
            else:
                reference_m = 3.0  # Dumpster
            
            fov_rad = np.radians(fov)
            height = (reference_m * w) / (2 * largest_px * np.tan(fov_rad / 2))
            
            return {
                'method': 'rectangle_detection',
                'height': height,
                'confidence': 0.4,
                'details': f'Found {len(rectangles)} rectangular objects'
            }
        
        return None
    
    def _estimate_from_detail_level(self, image: np.ndarray) -> Dict:
        """Estimate based on image detail/texture density."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Measure texture/detail using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # High detail = low altitude, low detail = high altitude
        # Typical ranges: very detailed (>1000) = 5-15m, medium (200-1000) = 15-50m, low (<200) = 50-100m+
        
        if laplacian_var > 1500:
            height = 8.0
        elif laplacian_var > 800:
            height = 20.0
        elif laplacian_var > 400:
            height = 35.0
        elif laplacian_var > 200:
            height = 50.0
        else:
            height = 80.0
        
        return {
            'method': 'detail_analysis',
            'height': height,
            'confidence': 0.3,
            'details': f'Laplacian variance: {laplacian_var:.0f}'
        }
    
    def _estimate_from_scene_coverage(self, image: np.ndarray) -> Dict:
        """Estimate based on what's visible in the scene."""
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Count different scene elements
        # Green (vegetation)
        green = cv2.inRange(hsv, (35, 30, 30), (85, 255, 255))
        green_pct = green.sum() / (h * w * 255)
        
        # Sky (if visible, very high altitude or tilted camera)
        sky = cv2.inRange(hsv, (100, 0, 180), (130, 80, 255))
        sky_pct = sky.sum() / (h * w * 255)
        
        # If lots of green visible (trees, grass), probably higher up
        # If no sky visible (looking straight down), typical drone altitude
        
        if sky_pct > 0.1:
            # Sky visible - either very high or tilted
            height = 60.0
            conf = 0.2
        elif green_pct > 0.4:
            # Lots of vegetation visible - medium-high
            height = 40.0
            conf = 0.35
        elif green_pct > 0.2:
            height = 25.0
            conf = 0.4
        else:
            # Mostly debris/industrial - probably closer
            height = 20.0
            conf = 0.4
        
        return {
            'method': 'scene_coverage',
            'height': height,
            'confidence': conf,
            'details': f'Green: {green_pct*100:.1f}%, Sky: {sky_pct*100:.1f}%'
        }
    
    def _get_height_category(self, height: float) -> str:
        """Get human-readable height category."""
        if height < 5:
            return 'pole_mounted'
        elif height < 15:
            return 'low_drone'
        elif height < 40:
            return 'medium_drone'
        elif height < 80:
            return 'high_drone'
        else:
            return 'aircraft'


def estimate_camera_height(image: np.ndarray) -> Dict:
    """Convenience function to estimate camera height."""
    estimator = CameraHeightEstimator()
    return estimator.estimate(image)
