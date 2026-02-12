"""
Pile segmentation using Segment Anything Model (SAM).
Isolates debris piles from background/ground in images.
"""

import numpy as np
from pathlib import Path
from typing import Optional
import cv2

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
    HAS_SAM = True
except ImportError:
    HAS_SAM = False


# Model paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
SAM_CHECKPOINT = MODELS_DIR / "sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"

# SAM download URL
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"


class PileSegmenter:
    """Segments debris piles from images using SAM or fallback methods."""
    
    def __init__(self, use_sam: bool = True, device: str = None):
        """
        Initialize the segmenter.
        
        Args:
            use_sam: Whether to use SAM (requires model download).
            device: 'cuda' or 'cpu'. Auto-detected if None.
        """
        self.use_sam = use_sam and HAS_SAM and HAS_TORCH
        self.device = device
        self.sam = None
        self.mask_generator = None
        
        if self.use_sam:
            self._init_sam()
    
    def _init_sam(self):
        """Initialize SAM model."""
        if not HAS_TORCH:
            print("[!] PyTorch not available, falling back to OpenCV segmentation")
            self.use_sam = False
            return
        
        if not HAS_SAM:
            print("[!] segment-anything not installed, falling back to OpenCV segmentation")
            self.use_sam = False
            return
            
        # Auto-detect device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Check if model exists
        if not SAM_CHECKPOINT.exists():
            print(f"[!] SAM checkpoint not found at {SAM_CHECKPOINT}")
            print(f"[*] Download from: {SAM_URL}")
            print(f"[*] Save to: {SAM_CHECKPOINT}")
            print("[*] Falling back to OpenCV segmentation")
            self.use_sam = False
            return
        
        print(f"[*] Loading SAM model on {self.device}...")
        self.sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT))
        self.sam.to(device=self.device)
        
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=16,  # Reduced for speed
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=1000,  # Filter tiny segments
        )
        print("[+] SAM model loaded")
    
    def segment(self, image: np.ndarray) -> dict:
        """
        Segment debris piles from an image.
        
        Args:
            image: BGR image (OpenCV format) or RGB.
        
        Returns:
            dict with:
                - 'mask': Binary mask of detected pile (H, W)
                - 'masks': List of individual segment masks
                - 'segments': List of segment info dicts
                - 'method': 'sam' or 'opencv'
        """
        if self.use_sam and self.mask_generator is not None:
            return self._segment_sam(image)
        else:
            return self._segment_opencv(image)
    
    def _segment_sam(self, image: np.ndarray) -> dict:
        """Segment using SAM automatic mask generation."""
        # SAM expects RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Generate masks
        masks = self.mask_generator.generate(image_rgb)
        
        # Sort by area (largest first)
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        # Combine top masks (likely the main pile)
        # Heuristic: take masks that are in the center/bottom of image (where piles usually are)
        h, w = image.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        segments = []
        for i, mask_info in enumerate(masks[:10]):  # Top 10 largest
            mask = mask_info['segmentation'].astype(np.uint8)
            
            # Calculate centroid
            moments = cv2.moments(mask)
            if moments['m00'] > 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
            else:
                cx, cy = w // 2, h // 2
            
            # Heuristic: prefer segments in lower 2/3 of image and not at edges
            is_pile_candidate = (
                cy > h * 0.3 and  # Not in top 30%
                cx > w * 0.1 and cx < w * 0.9 and  # Not at left/right edges
                mask_info['area'] > (h * w * 0.01)  # At least 1% of image
            )
            
            segments.append({
                'id': i,
                'area': mask_info['area'],
                'centroid': (cx, cy),
                'bbox': mask_info['bbox'],
                'is_pile_candidate': is_pile_candidate,
                'stability_score': mask_info['stability_score'],
            })
            
            if is_pile_candidate:
                combined_mask = np.maximum(combined_mask, mask * 255)
        
        return {
            'mask': combined_mask,
            'masks': [m['segmentation'] for m in masks[:10]],
            'segments': segments,
            'method': 'sam',
        }
    
    def _segment_opencv(self, image: np.ndarray) -> dict:
        """
        Improved segmentation focusing on containers and debris piles.
        Strategy: Find textured/complex regions, exclude obvious non-debris (sky, vegetation).
        """
        h, w = image.shape[:2]
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # === STRICT EXCLUSIONS ===
        # Exclude sky (bright blue/white)
        sky_mask = cv2.inRange(hsv, (100, 0, 200), (130, 60, 255))
        # Exclude vegetation (green)
        green_mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
        # Exclude bright uniform areas (buildings, roads)
        bright_uniform = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
        exclude_mask = cv2.bitwise_or(cv2.bitwise_or(sky_mask, green_mask), bright_uniform)
        exclude_mask = cv2.dilate(exclude_mask, np.ones((15, 15), np.uint8))
        
        # === FIND DEBRIS PILES ===
        # Debris piles have: high texture variance, mixed colors, irregular shapes
        # Texture variance (high variance = debris, low = uniform buildings)
        texture = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
        texture_blur = cv2.GaussianBlur(texture, (15, 15), 0)
        texture_var = cv2.GaussianBlur((texture - texture_blur) ** 2, (15, 15), 0)
        texture_norm = (texture_var / (texture_var.max() + 1e-8) * 255).astype(np.uint8)
        _, textured = cv2.threshold(texture_norm, 30, 255, cv2.THRESH_BINARY)  # Higher threshold
        
        # Color variance - debris has mixed colors (high variance)
        # Buildings/roads are uniform (low variance)
        b, g, r = cv2.split(image)
        color_var = np.sqrt((b.astype(float) - gray.astype(float))**2 + 
                           (g.astype(float) - gray.astype(float))**2 + 
                           (r.astype(float) - gray.astype(float))**2)
        color_var_norm = (color_var / (color_var.max() + 1e-8) * 255).astype(np.uint8)
        _, high_var = cv2.threshold(color_var_norm, 40, 255, cv2.THRESH_BINARY)
        
        # Combine: high texture AND high color variance = debris
        combined = cv2.bitwise_and(textured, high_var)
        
        # Remove exclusions (buildings, sky, vegetation)
        exclude_mask = cv2.bitwise_or(exclude_mask, cv2.inRange(hsv, (0, 0, 200), (180, 30, 255)))  # Bright uniform
        exclude_mask = cv2.dilate(exclude_mask, np.ones((15, 15), np.uint8))
        combined = cv2.bitwise_and(combined, cv2.bitwise_not(exclude_mask))
        
        # Morphological cleanup - larger kernels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_small)
        
        # Fill holes
        contours_fill, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(filled, contours_fill, -1, 255, -1)
        
        # Find and STRICTLY filter contours
        contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        final_mask = np.zeros((h, w), dtype=np.uint8)
        segments = []
        
        # Stricter area limits - debris piles are typically 2-30% of image
        min_area = h * w * 0.02   # At least 2% of image
        max_area = h * w * 0.30   # Max 30% (buildings are larger)
        
        for i, contour in enumerate(sorted(contours, key=cv2.contourArea, reverse=True)[:15]):
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            
            x, y, bw, bh = cv2.boundingRect(contour)
            moments = cv2.moments(contour)
            if moments['m00'] > 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
            else:
                cx, cy = x + bw // 2, y + bh // 2
            
            aspect = max(bw, bh) / (min(bw, bh) + 1)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / (hull_area + 1)
            
            # Check texture/variance in this region
            roi = image[y:y+bh, x:x+bw]
            if roi.size > 0:
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi_texture = np.std(cv2.Laplacian(roi_gray, cv2.CV_64F))
                roi_color_var = np.std(roi.reshape(-1, 3).astype(float), axis=0).mean()
            else:
                roi_texture = 0
                roi_color_var = 0
            
            # STRICT filtering: exclude buildings (perfect rectangles), signs (small), roads (elongated)
            is_pile_candidate = (
                aspect < 5 and  # Not too elongated (roads)
                solidity > 0.3 and solidity < 0.9 and  # Irregular, not perfect rectangle (buildings)
                roi_texture > 20 and  # High texture (debris)
                roi_color_var > 15  # Mixed colors (debris)
            )
            
            segments.append({
                'id': i,
                'area': int(area),
                'centroid': (cx, cy),
                'bbox': [x, y, bw, bh],
                'aspect_ratio': round(aspect, 2),
                'solidity': round(solidity, 2),
                'texture': round(roi_texture, 1),
                'color_var': round(roi_color_var, 1),
                'is_pile_candidate': is_pile_candidate,
            })
            
            if is_pile_candidate:
                cv2.drawContours(final_mask, [contour], -1, 255, -1)
        
        return {
            'mask': final_mask,
            'masks': [final_mask],
            'segments': segments,
            'method': 'opencv_v3_strict',
        }
    
    def visualize(self, image: np.ndarray, result: dict, alpha: float = 0.5) -> np.ndarray:
        """
        Create visualization overlay of segmentation.
        
        Args:
            image: Original BGR image.
            result: Output from segment().
            alpha: Overlay transparency.
        
        Returns:
            BGR image with mask overlay.
        """
        vis = image.copy()
        mask = result['mask']
        
        # Create colored overlay
        overlay = np.zeros_like(image)
        overlay[mask > 0] = [0, 255, 0]  # Green for pile
        
        # Blend
        vis = cv2.addWeighted(vis, 1, overlay, alpha, 0)
        
        # Draw contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
        
        # Add method label
        cv2.putText(vis, f"Method: {result['method']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return vis


def segment_image(image_path: str | Path, use_sam: bool = True) -> dict:
    """
    Convenience function to segment a single image.
    
    Args:
        image_path: Path to image file.
        use_sam: Whether to try SAM (falls back to OpenCV if unavailable).
    
    Returns:
        Segmentation result dict.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    segmenter = PileSegmenter(use_sam=use_sam)
    return segmenter.segment(image)


if __name__ == "__main__":
    # Test with sample image if available
    from .scraper import list_available_images, DATA_DIR
    
    images = list_available_images()
    if images:
        print(f"[*] Testing segmentation on: {images[0]}")
        
        segmenter = PileSegmenter(use_sam=True)
        image = cv2.imread(str(images[0]))
        result = segmenter.segment(image)
        
        print(f"[+] Method: {result['method']}")
        print(f"[+] Segments found: {len(result['segments'])}")
        for seg in result['segments']:
            print(f"    - ID {seg['id']}: area={seg['area']}, pile={seg.get('is_pile_candidate', 'N/A')}")
        
        # Save visualization
        vis = segmenter.visualize(image, result)
        output_path = DATA_DIR / "segmentation_test.jpg"
        cv2.imwrite(str(output_path), vis)
        print(f"[+] Saved visualization to: {output_path}")
    else:
        print("[!] No images available. Run scraper.py first.")
