"""
Change detection for tracking pile growth/shrinkage over time.
Compares two images to identify changes in debris piles.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import cv2

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


class ChangeDetector:
    """Detects changes between two images of debris piles."""
    
    def __init__(
        self,
        method: str = 'combined',
        threshold: float = 0.1,
    ):
        """
        Initialize change detector.
        
        Args:
            method: Detection method - 'pixel', 'ssim', 'feature', or 'combined'.
            threshold: Sensitivity threshold (0-1). Lower = more sensitive.
        """
        self.method = method
        self.threshold = threshold
    
    def detect(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        mask1: np.ndarray = None,
        mask2: np.ndarray = None,
    ) -> Dict:
        """
        Detect changes between two images.
        
        Args:
            image1: First (earlier) BGR image.
            image2: Second (later) BGR image.
            mask1: Optional mask for image1 (pile region).
            mask2: Optional mask for image2 (pile region).
        
        Returns:
            Dict with change metrics and visualizations.
        """
        # Ensure same size
        if image1.shape != image2.shape:
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
        
        h, w = image1.shape[:2]
        
        # Convert to grayscale for comparison
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        
        # Calculate changes using selected method
        if self.method == 'pixel':
            result = self._detect_pixel(gray1, gray2)
        elif self.method == 'ssim':
            result = self._detect_ssim(gray1, gray2)
        elif self.method == 'feature':
            result = self._detect_feature(image1, image2)
        else:  # combined
            result = self._detect_combined(gray1, gray2, image1, image2)
        
        # Analyze mask changes if provided
        if mask1 is not None and mask2 is not None:
            mask_result = self._analyze_mask_change(mask1, mask2)
            result.update(mask_result)
        
        # Create visualization
        result['visualization'] = self._create_visualization(
            image1, image2, result.get('change_mask', np.zeros((h, w), dtype=np.uint8))
        )
        
        # Classify change type
        result['change_type'] = self._classify_change(result)
        
        return result
    
    def _detect_pixel(self, gray1: np.ndarray, gray2: np.ndarray) -> Dict:
        """Pixel-wise difference detection."""
        # Absolute difference
        diff = cv2.absdiff(gray1, gray2)
        
        # Apply Gaussian blur to reduce noise
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        
        # Threshold
        thresh_val = int(255 * self.threshold)
        _, change_mask = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_OPEN, kernel)
        change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_CLOSE, kernel)
        
        # Calculate metrics
        total_pixels = gray1.size
        changed_pixels = np.count_nonzero(change_mask)
        change_percentage = changed_pixels / total_pixels
        
        return {
            'method': 'pixel',
            'change_mask': change_mask,
            'diff_map': diff,
            'change_percentage': change_percentage,
            'changed_pixels': changed_pixels,
            'total_pixels': total_pixels,
            'mean_diff': float(diff.mean()),
            'max_diff': float(diff.max()),
        }
    
    def _detect_ssim(self, gray1: np.ndarray, gray2: np.ndarray) -> Dict:
        """Structural similarity based detection."""
        if not HAS_SKIMAGE:
            print("[!] scikit-image not available, falling back to pixel method")
            return self._detect_pixel(gray1, gray2)
        
        # Calculate SSIM with full output
        score, diff = ssim(gray1, gray2, full=True)
        
        # Convert diff to uint8
        diff = ((1 - diff) * 255).astype(np.uint8)
        
        # Threshold
        thresh_val = int(255 * self.threshold * 2)  # SSIM needs higher threshold
        _, change_mask = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)
        
        # Cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_OPEN, kernel)
        
        total_pixels = gray1.size
        changed_pixels = np.count_nonzero(change_mask)
        
        return {
            'method': 'ssim',
            'change_mask': change_mask,
            'diff_map': diff,
            'ssim_score': float(score),
            'change_percentage': changed_pixels / total_pixels,
            'changed_pixels': changed_pixels,
            'total_pixels': total_pixels,
        }
    
    def _detect_feature(self, image1: np.ndarray, image2: np.ndarray) -> Dict:
        """Feature-based detection using ORB."""
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        
        # Create ORB detector
        orb = cv2.ORB_create(nfeatures=1000)
        
        # Detect keypoints and compute descriptors
        kp1, desc1 = orb.detectAndCompute(gray1, None)
        kp2, desc2 = orb.detectAndCompute(gray2, None)
        
        if desc1 is None or desc2 is None:
            # No features found, fall back to pixel
            return self._detect_pixel(gray1, gray2)
        
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Calculate match statistics
        good_matches = [m for m in matches if m.distance < 50]
        match_ratio = len(good_matches) / max(len(kp1), len(kp2), 1)
        
        # Create change mask from unmatched regions
        h, w = gray1.shape
        change_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Areas with many unmatched keypoints suggest change
        matched_pts = set()
        for m in good_matches:
            pt = kp1[m.queryIdx].pt
            matched_pts.add((int(pt[0]), int(pt[1])))
        
        for kp in kp1:
            pt = (int(kp.pt[0]), int(kp.pt[1]))
            if pt not in matched_pts:
                cv2.circle(change_mask, pt, 20, 255, -1)
        
        # Cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_CLOSE, kernel)
        
        changed_pixels = np.count_nonzero(change_mask)
        
        return {
            'method': 'feature',
            'change_mask': change_mask,
            'diff_map': cv2.absdiff(gray1, gray2),
            'match_ratio': match_ratio,
            'total_features': len(kp1),
            'matched_features': len(good_matches),
            'change_percentage': changed_pixels / gray1.size,
            'changed_pixels': changed_pixels,
            'total_pixels': gray1.size,
        }
    
    def _detect_combined(
        self,
        gray1: np.ndarray,
        gray2: np.ndarray,
        image1: np.ndarray,
        image2: np.ndarray,
    ) -> Dict:
        """Combine multiple detection methods."""
        # Get results from each method
        pixel_result = self._detect_pixel(gray1, gray2)
        ssim_result = self._detect_ssim(gray1, gray2)
        
        # Combine masks (intersection for higher confidence)
        combined_mask = cv2.bitwise_or(
            pixel_result['change_mask'],
            ssim_result['change_mask']
        )
        
        # Cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        changed_pixels = np.count_nonzero(combined_mask)
        total_pixels = gray1.size
        
        return {
            'method': 'combined',
            'change_mask': combined_mask,
            'diff_map': pixel_result['diff_map'],
            'change_percentage': changed_pixels / total_pixels,
            'changed_pixels': changed_pixels,
            'total_pixels': total_pixels,
            'ssim_score': ssim_result.get('ssim_score', 0),
            'pixel_methods': {
                'pixel_change': pixel_result['change_percentage'],
                'ssim_change': ssim_result['change_percentage'],
            },
        }
    
    def _analyze_mask_change(self, mask1: np.ndarray, mask2: np.ndarray) -> Dict:
        """Analyze change in pile masks between images."""
        # Ensure same size and binary
        if mask1.shape != mask2.shape:
            mask2 = cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]))
        
        mask1_bin = (mask1 > 0).astype(np.uint8)
        mask2_bin = (mask2 > 0).astype(np.uint8)
        
        # Calculate areas
        area1 = mask1_bin.sum()
        area2 = mask2_bin.sum()
        
        # Calculate growth/shrinkage
        added = cv2.bitwise_and(mask2_bin, cv2.bitwise_not(mask1_bin))  # New areas
        removed = cv2.bitwise_and(mask1_bin, cv2.bitwise_not(mask2_bin))  # Gone areas
        
        added_pixels = added.sum()
        removed_pixels = removed.sum()
        
        # Net change
        net_change = area2 - area1
        net_change_pct = net_change / (area1 + 1e-8)
        
        return {
            'mask_area_before': int(area1),
            'mask_area_after': int(area2),
            'mask_added_pixels': int(added_pixels),
            'mask_removed_pixels': int(removed_pixels),
            'mask_net_change': int(net_change),
            'mask_net_change_pct': float(net_change_pct),
            'pile_grew': net_change > 0,
            'pile_shrank': net_change < 0,
        }
    
    def _classify_change(self, result: Dict) -> str:
        """Classify the type of change detected."""
        change_pct = result.get('change_percentage', 0)
        mask_change = result.get('mask_net_change_pct', 0)
        
        if change_pct < 0.01:
            return 'no_change'
        elif change_pct < 0.05:
            return 'minor_change'
        elif mask_change > 0.2:
            return 'significant_growth'
        elif mask_change < -0.2:
            return 'significant_reduction'
        elif change_pct > 0.3:
            return 'major_change'
        else:
            return 'moderate_change'
    
    def _create_visualization(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        change_mask: np.ndarray,
    ) -> np.ndarray:
        """Create side-by-side visualization with change overlay."""
        h, w = image1.shape[:2]
        
        # Create comparison image
        comparison = np.zeros((h, w * 2 + 10, 3), dtype=np.uint8)
        comparison[:, :w] = image1
        comparison[:, w+10:] = image2
        comparison[:, w:w+10] = 128  # Gray separator
        
        # Add change overlay on second image
        overlay = image2.copy()
        # Red for changes
        overlay[change_mask > 0] = [0, 0, 255]
        blended = cv2.addWeighted(image2, 0.7, overlay, 0.3, 0)
        comparison[:, w+10:] = blended
        
        # Add labels
        cv2.putText(comparison, "Before", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "After (changes in red)", (w + 20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return comparison


def detect_changes(
    image1: np.ndarray,
    image2: np.ndarray,
    threshold: float = 0.1,
) -> Dict:
    """
    Convenience function for change detection.
    
    Args:
        image1: First (earlier) image.
        image2: Second (later) image.
        threshold: Sensitivity threshold.
    
    Returns:
        Change detection results.
    """
    detector = ChangeDetector(threshold=threshold)
    return detector.detect(image1, image2)


def get_change_summary(result: Dict) -> str:
    """Get human-readable summary of changes."""
    lines = [
        f"Change Analysis:",
        f"  Change Type: {result.get('change_type', 'unknown')}",
        f"  Changed Area: {result.get('change_percentage', 0)*100:.1f}%",
    ]
    
    if 'ssim_score' in result:
        lines.append(f"  Similarity Score: {result['ssim_score']:.3f}")
    
    if 'mask_net_change_pct' in result:
        pct = result['mask_net_change_pct'] * 100
        direction = "grew" if pct > 0 else "shrank"
        lines.append(f"  Pile {direction} by: {abs(pct):.1f}%")
    
    return "\n".join(lines)


if __name__ == "__main__":
    from .scraper import list_available_images, DATA_DIR
    
    images = list_available_images()
    if len(images) >= 2:
        print(f"[*] Testing change detection between:")
        print(f"    Image 1: {images[0]}")
        print(f"    Image 2: {images[1]}")
        
        detector = ChangeDetector()
        img1 = cv2.imread(str(images[0]))
        img2 = cv2.imread(str(images[1]))
        
        result = detector.detect(img1, img2)
        
        print(get_change_summary(result))
        
        # Save visualization
        output_path = DATA_DIR / "change_detection_test.jpg"
        cv2.imwrite(str(output_path), result['visualization'])
        print(f"\n[+] Saved visualization to: {output_path}")
    else:
        print("[!] Need at least 2 images. Run scraper.py first.")
