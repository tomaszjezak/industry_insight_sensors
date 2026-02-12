"""
Container and pallet detection for recycling yards.
Uses SAM + heuristics to detect shipping containers, pallets, and bales.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import cv2

from .segmentation import PileSegmenter


# Standard container/pallet dimensions (for size estimation)
CONTAINER_SIZES = {
    '20ft': {'length_m': 6.1, 'width_m': 2.4, 'height_m': 2.6},
    '40ft': {'length_m': 12.2, 'width_m': 2.4, 'height_m': 2.6},
}

PALLET_SIZES = {
    'standard': {'length_m': 1.2, 'width_m': 1.0, 'height_m': 1.5},  # Typical bale height
}


class ContainerDetector:
    """
    Detects shipping containers, pallets, and bales in images.
    Uses SAM for segmentation + heuristics for shape/size matching.
    """
    
    def __init__(self, use_sam: bool = True, device: str = None):
        """
        Initialize container detector.
        
        Args:
            use_sam: Whether to use SAM for segmentation.
            device: 'cuda' or 'cpu'. Auto-detected if None.
        """
        self.segmenter = PileSegmenter(use_sam=use_sam, device=device)
    
    def detect(
        self,
        image: np.ndarray,
        camera_height_m: float = 30.0,
        fov_degrees: float = 70.0,
    ) -> Dict:
        """
        Detect containers, pallets, and bales in image.
        
        Args:
            image: BGR image
            camera_height_m: Camera height for size estimation
            fov_degrees: Camera field of view
        
        Returns:
            Dict with detected objects and counts
        """
        h, w = image.shape[:2]
        
        # Get segmentation
        seg_result = self.segmenter.segment(image)
        mask = seg_result['mask']
        
        # Find rectangular objects (containers, pallets)
        containers = self._detect_containers(image, mask, camera_height_m, fov_degrees)
        pallets = self._detect_pallets(image, mask, camera_height_m, fov_degrees)
        
        return {
            'containers': containers,
            'pallets': pallets,
            'container_count': len(containers),
            'pallet_count': len(pallets),
            'total_objects': len(containers) + len(pallets),
        }
    
    def _detect_containers(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        camera_height_m: float,
        fov_degrees: float,
    ) -> List[Dict]:
        """
        Detect shipping containers using shape and size heuristics.
        
        Containers are:
        - Rectangular (aspect ratio ~2.5:1 for 40ft, ~2.5:1 for 20ft)
        - Large (visible from aerial view)
        - Often have distinctive colors (red, blue, white, yellow)
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find contours in mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        containers = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:  # Too small
                continue
            
            # Get bounding rectangle
            rect = cv2.minAreaRect(contour)
            box_w, box_h = rect[1]
            angle = rect[2]
            
            # Container-like aspect ratio (2:1 to 3:1)
            aspect = max(box_w, box_h) / (min(box_w, box_h) + 1)
            if not (1.8 < aspect < 3.5):
                continue
            
            # Estimate real-world size
            fov_rad = np.radians(fov_degrees)
            visible_width_m = 2 * camera_height_m * np.tan(fov_rad / 2)
            m_per_pixel = visible_width_m / w
            
            length_px = max(box_w, box_h)
            length_m = length_px * m_per_pixel
            
            # Check if size matches container dimensions
            # 20ft = 6.1m, 40ft = 12.2m (with some tolerance)
            container_type = None
            if 5.0 < length_m < 7.5:
                container_type = '20ft'
            elif 10.0 < length_m < 14.0:
                container_type = '40ft'
            elif 7.5 < length_m < 10.0:
                container_type = 'unknown'  # Could be 20ft or partial view
            
            if container_type:
                # Get bounding box
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                # Get centroid
                moments = cv2.moments(contour)
                if moments['m00'] > 0:
                    cx = int(moments['m10'] / moments['m00'])
                    cy = int(moments['m01'] / moments['m00'])
                else:
                    cx, cy = int(rect[0][0]), int(rect[0][1])
                
                containers.append({
                    'type': container_type,
                    'bbox': box.tolist(),
                    'centroid': (cx, cy),
                    'length_m': round(length_m, 2),
                    'width_m': round(min(box_w, box_h) * m_per_pixel, 2),
                    'area_px': int(area),
                    'confidence': 0.7 if container_type != 'unknown' else 0.5,
                })
        
        return containers
    
    def _detect_pallets(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        camera_height_m: float,
        fov_degrees: float,
    ) -> List[Dict]:
        """
        Detect pallets/bales using shape and size heuristics.
        
        Pallets/bales are:
        - Square-ish or rectangular (aspect ratio ~1:1 to 1.5:1)
        - Smaller than containers
        - Often stacked
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        pallets = []
        
        # Estimate pixel-to-meter conversion
        fov_rad = np.radians(fov_degrees)
        visible_width_m = 2 * camera_height_m * np.tan(fov_rad / 2)
        m_per_pixel = visible_width_m / w
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 200:  # Too small
                continue
            
            # Get bounding rectangle
            rect = cv2.minAreaRect(contour)
            box_w, box_h = rect[1]
            
            # Pallet-like aspect ratio (1:1 to 1.5:1)
            aspect = max(box_w, box_h) / (min(box_w, box_h) + 1)
            if aspect > 2.0:  # Too elongated, probably not a pallet
                continue
            
            # Estimate size
            size_px = max(box_w, box_h)
            size_m = size_px * m_per_pixel
            
            # Pallet/bale size range: 0.8m to 2.5m
            if 0.8 < size_m < 2.5:
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                moments = cv2.moments(contour)
                if moments['m00'] > 0:
                    cx = int(moments['m10'] / moments['m00'])
                    cy = int(moments['m01'] / moments['m00'])
                else:
                    cx, cy = int(rect[0][0]), int(rect[0][1])
                
                pallets.append({
                    'bbox': box.tolist(),
                    'centroid': (cx, cy),
                    'size_m': round(size_m, 2),
                    'area_px': int(area),
                    'confidence': 0.6,
                })
        
        return pallets
    
    def visualize(
        self,
        image: np.ndarray,
        result: Dict,
    ) -> np.ndarray:
        """
        Create visualization of detected containers and pallets.
        
        Args:
            image: Original BGR image
            result: Output from detect()
        
        Returns:
            BGR image with annotations
        """
        vis = image.copy()
        
        # Draw containers (red)
        for container in result['containers']:
            bbox = np.array(container['bbox'], dtype=np.int32)
            cv2.drawContours(vis, [bbox], -1, (0, 0, 255), 2)
            cx, cy = container['centroid']
            label = f"{container['type']} ({container['length_m']}m)"
            cv2.putText(vis, label, (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw pallets (green)
        for pallet in result['pallets']:
            bbox = np.array(pallet['bbox'], dtype=np.int32)
            cv2.drawContours(vis, [bbox], -1, (0, 255, 0), 2)
            cx, cy = pallet['centroid']
            label = f"Pallet ({pallet['size_m']}m)"
            cv2.putText(vis, label, (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Add summary text
        summary = f"Containers: {result['container_count']}, Pallets: {result['pallet_count']}"
        cv2.putText(vis, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis


def detect_containers(
    image: np.ndarray,
    camera_height_m: float = 30.0,
) -> Dict:
    """
    Convenience function for container detection.
    
    Args:
        image: BGR image
        camera_height_m: Camera height
    
    Returns:
        Detection results
    """
    detector = ContainerDetector()
    return detector.detect(image, camera_height_m=camera_height_m)
