"""
Volume calculation from depth map and segmentation mask.
Estimates pile volume in cubic meters using camera geometry.
"""

import numpy as np
from typing import Tuple, Dict
import cv2


class VolumeCalculator:
    """Calculates pile volume from depth map and segmentation mask."""
    
    def __init__(
        self,
        camera_height_m: float = 3.0,
        fov_horizontal_deg: float = 90.0,
        fov_vertical_deg: float = 65.0,
    ):
        """
        Initialize volume calculator with camera parameters.
        
        Args:
            camera_height_m: Height of camera above ground in meters.
            fov_horizontal_deg: Horizontal field of view in degrees.
            fov_vertical_deg: Vertical field of view in degrees.
        """
        self.camera_height_m = camera_height_m
        self.fov_h = np.radians(fov_horizontal_deg)
        self.fov_v = np.radians(fov_vertical_deg)
    
    def calculate(
        self,
        depth_normalized: np.ndarray,
        mask: np.ndarray,
        image_shape: Tuple[int, int] = None,
    ) -> Dict:
        """
        Calculate volume of pile from depth map and mask.
        
        The method:
        1. Convert pixel area to real-world area using FOV and camera height
        2. Estimate height at each pixel from depth
        3. Integrate (sum) height * area for all pile pixels
        
        Args:
            depth_normalized: Normalized depth map (0-1), higher = closer/taller.
            mask: Binary mask of pile area.
            image_shape: (height, width) of original image. Inferred if None.
        
        Returns:
            Dict with volume and related metrics.
        """
        if image_shape is None:
            image_shape = depth_normalized.shape[:2]
        
        h, w = image_shape
        
        # Calculate ground footprint visible in image
        # For overhead camera at height H with FOV, visible area is:
        # width_m = 2 * H * tan(fov_h / 2)
        # height_m = 2 * H * tan(fov_v / 2)
        visible_width_m = 2 * self.camera_height_m * np.tan(self.fov_h / 2)
        visible_height_m = 2 * self.camera_height_m * np.tan(self.fov_v / 2)
        
        # Meters per pixel
        m_per_pixel_x = visible_width_m / w
        m_per_pixel_y = visible_height_m / h
        pixel_area_m2 = m_per_pixel_x * m_per_pixel_y
        
        # Binary mask
        mask_binary = (mask > 0).astype(float)
        
        # Count pile pixels
        pile_pixels = mask_binary.sum()
        if pile_pixels == 0:
            return {
                'volume_m3': 0.0,
                'pile_area_m2': 0.0,
                'avg_height_m': 0.0,
                'max_height_m': 0.0,
                'min_height_m': 0.0,
                'std_height_m': 0.0,
                'pixel_count': 0,
                'camera_height_m': self.camera_height_m,
                'visible_area_m2': visible_width_m * visible_height_m,
                'm_per_pixel': (m_per_pixel_x + m_per_pixel_y) / 2,
            }
        
        # Pile area in m²
        pile_area_m2 = pile_pixels * pixel_area_m2
        
        # Estimate ground plane (lowest depth in surrounding area)
        # Dilate mask to get surrounding region
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
        dilated = cv2.dilate(mask_binary.astype(np.uint8), kernel)
        surrounding = (dilated > 0) & (mask_binary == 0)
        
        if surrounding.any():
            ground_depth = np.percentile(depth_normalized[surrounding], 25)
        else:
            # Use bottom 10% of depth values as ground reference
            ground_depth = np.percentile(depth_normalized, 10)
        
        # Height map: difference from ground, scaled by camera height
        # Clamp to reasonable range (0 to camera_height)
        height_map = (depth_normalized - ground_depth) * self.camera_height_m
        height_map = np.clip(height_map, 0, self.camera_height_m * 0.9)  # Max 90% of camera height
        
        # Apply mask
        pile_heights = height_map[mask_binary > 0]
        
        # Calculate statistics
        avg_height_m = float(pile_heights.mean())
        max_height_m = float(pile_heights.max())
        min_height_m = float(pile_heights.min())
        std_height_m = float(pile_heights.std())
        
        # Volume = integral of height over area
        # For discrete pixels: sum(height_i * pixel_area)
        volume_m3 = float((height_map * mask_binary).sum() * pixel_area_m2)
        
        return {
            'volume_m3': volume_m3,
            'pile_area_m2': pile_area_m2,
            'avg_height_m': avg_height_m,
            'max_height_m': max_height_m,
            'min_height_m': min_height_m,
            'std_height_m': std_height_m,
            'pixel_count': int(pile_pixels),
            'camera_height_m': self.camera_height_m,
            'visible_area_m2': visible_width_m * visible_height_m,
            'm_per_pixel': (m_per_pixel_x + m_per_pixel_y) / 2,
        }
    
    def estimate_tonnage(
        self,
        volume_m3: float,
        material_type: str = 'mixed',
    ) -> Dict:
        """
        Estimate weight from volume using typical C&D material densities.
        
        Args:
            volume_m3: Volume in cubic meters.
            material_type: One of 'concrete', 'wood', 'metal', 'mixed', 'soil'.
        
        Returns:
            Dict with tonnage estimates.
        """
        # Typical loose bulk densities for C&D materials (tons/m³)
        # Note: These are for loose piles, not compacted
        DENSITIES = {
            'concrete': 1.4,    # Broken concrete: 1.2-1.6 t/m³
            'wood': 0.35,       # Loose wood debris: 0.2-0.5 t/m³
            'metal': 0.8,       # Mixed scrap metal: 0.5-1.2 t/m³
            'soil': 1.5,        # Loose soil/dirt: 1.3-1.7 t/m³
            'mixed': 0.9,       # Mixed C&D: 0.6-1.2 t/m³
            'drywall': 0.4,     # Drywall/gypsum: 0.3-0.5 t/m³
            'asphalt': 1.3,     # Broken asphalt: 1.2-1.5 t/m³
        }
        
        density = DENSITIES.get(material_type, DENSITIES['mixed'])
        tonnage = volume_m3 * density
        
        return {
            'tonnage': tonnage,
            'density_used': density,
            'material_type': material_type,
            'tonnage_range': {
                'low': volume_m3 * (density * 0.7),
                'high': volume_m3 * (density * 1.3),
            }
        }
    
    def create_height_visualization(
        self,
        depth_normalized: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Create a height map visualization.
        
        Args:
            depth_normalized: Normalized depth map.
            mask: Binary pile mask.
        
        Returns:
            BGR visualization image.
        """
        # Get ground plane
        mask_binary = (mask > 0).astype(float)
        non_pile = mask_binary == 0
        
        if non_pile.any():
            ground_depth = np.percentile(depth_normalized[non_pile], 25)
        else:
            ground_depth = np.percentile(depth_normalized, 10)
        
        # Height map
        height_map = (depth_normalized - ground_depth)
        height_map = np.clip(height_map, 0, 1)
        
        # Apply mask
        height_map = height_map * mask_binary
        
        # Colorize
        height_uint8 = (height_map * 255).astype(np.uint8)
        colored = cv2.applyColorMap(height_uint8, cv2.COLORMAP_JET)
        
        # Make background gray
        colored[mask == 0] = [50, 50, 50]
        
        return colored


def calculate_volume(
    depth_normalized: np.ndarray,
    mask: np.ndarray,
    camera_height_m: float = 3.0,
) -> Dict:
    """
    Convenience function for volume calculation.
    
    Args:
        depth_normalized: Normalized depth map (0-1).
        mask: Binary pile mask.
        camera_height_m: Camera height in meters.
    
    Returns:
        Volume calculation results.
    """
    calculator = VolumeCalculator(camera_height_m=camera_height_m)
    return calculator.calculate(depth_normalized, mask)


if __name__ == "__main__":
    # Test with synthetic data
    print("[*] Testing volume calculation with synthetic data...")
    
    # Create synthetic depth map (hemisphere pile)
    size = 256
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    r = np.sqrt(x**2 + y**2)
    
    # Hemisphere
    depth = np.zeros((size, size))
    pile_region = r < 0.5
    depth[pile_region] = np.sqrt(0.25 - r[pile_region]**2)
    
    # Normalize
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    
    # Create mask
    mask = (r < 0.5).astype(np.uint8) * 255
    
    # Calculate volume
    calc = VolumeCalculator(camera_height_m=3.0)
    result = calc.calculate(depth_norm, mask)
    
    print(f"[+] Results:")
    print(f"    Volume: {result['volume_m3']:.2f} m³")
    print(f"    Area: {result['pile_area_m2']:.2f} m²")
    print(f"    Avg Height: {result['avg_height_m']:.2f} m")
    print(f"    Max Height: {result['max_height_m']:.2f} m")
    
    # Tonnage estimate
    tonnage = calc.estimate_tonnage(result['volume_m3'], 'mixed')
    print(f"    Est. Tonnage: {tonnage['tonnage']:.1f} tons (mixed debris)")
