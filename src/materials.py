"""
Material classification for C&D debris using color analysis.
Identifies concrete, wood, metal, and mixed materials.
"""

import numpy as np
from typing import Dict, Tuple, List
import cv2


class MaterialClassifier:
    """Classifies C&D materials based on color analysis."""
    
    # HSV color ranges for different materials (H: 0-180, S: 0-255, V: 0-255)
    # These are approximate and work for typical C&D debris
    MATERIAL_COLORS = {
        'concrete': {
            # Gray tones - low saturation, variable value
            'ranges': [
                ((0, 0, 80), (180, 50, 200)),     # Light gray
                ((0, 0, 40), (180, 40, 120)),     # Dark gray
            ],
            'color_bgr': (180, 180, 180),  # Display color
        },
        'wood': {
            # Brown/tan tones
            'ranges': [
                ((8, 40, 60), (25, 200, 220)),    # Light brown/tan
                ((0, 50, 40), (12, 180, 180)),    # Reddish brown
                ((25, 30, 80), (35, 150, 200)),   # Yellow-brown
            ],
            'color_bgr': (60, 120, 180),  # Display color (tan)
        },
        'metal': {
            # Metallic - often high value, variable saturation
            # Rust: orange-red
            'ranges': [
                ((0, 100, 100), (15, 255, 255)),    # Rust (red-orange)
                ((15, 100, 100), (25, 255, 255)),   # Rust (orange)
                ((0, 0, 180), (180, 30, 255)),      # Shiny metal (high V, low S)
            ],
            'color_bgr': (100, 100, 200),  # Display color (rusty)
        },
        'vegetation': {
            # Green - to exclude from debris
            'ranges': [
                ((35, 40, 40), (85, 255, 255)),   # Green
            ],
            'color_bgr': (0, 200, 0),
        },
        'plastic': {
            # Various bright colors
            'ranges': [
                ((90, 100, 100), (130, 255, 255)),  # Blue plastic
                ((0, 150, 150), (10, 255, 255)),    # Red plastic
                ((20, 150, 150), (35, 255, 255)),   # Yellow plastic
            ],
            'color_bgr': (200, 100, 100),
        },
    }
    
    def __init__(self, materials: List[str] = None):
        """
        Initialize classifier.
        
        Args:
            materials: List of materials to classify. 
                      Default: ['concrete', 'wood', 'metal', 'mixed']
        """
        self.materials = materials or ['concrete', 'wood', 'metal']
    
    def classify(
        self,
        image: np.ndarray,
        mask: np.ndarray = None,
    ) -> Dict:
        """
        Classify materials in an image.
        
        Args:
            image: BGR image.
            mask: Optional binary mask to restrict analysis.
        
        Returns:
            Dict with material percentages and visualization.
        """
        h, w = image.shape[:2]
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create analysis mask
        if mask is not None:
            analysis_mask = (mask > 0).astype(np.uint8)
        else:
            analysis_mask = np.ones((h, w), dtype=np.uint8)
        
        total_pixels = analysis_mask.sum()
        if total_pixels == 0:
            return self._empty_result()
        
        # Classify each material
        material_masks = {}
        material_counts = {}
        
        for material in self.materials:
            if material not in self.MATERIAL_COLORS:
                continue
            
            material_mask = self._create_material_mask(hsv, material)
            material_mask = cv2.bitwise_and(material_mask, material_mask, mask=analysis_mask)
            
            material_masks[material] = material_mask
            material_counts[material] = material_mask.sum() // 255  # Count white pixels
        
        # Calculate unclassified (mixed/other)
        classified_mask = np.zeros((h, w), dtype=np.uint8)
        for mat_mask in material_masks.values():
            classified_mask = cv2.bitwise_or(classified_mask, mat_mask)
        
        mixed_count = total_pixels - (classified_mask.sum() // 255)
        material_counts['mixed'] = max(0, mixed_count)
        
        # Calculate percentages
        percentages = {}
        for material, count in material_counts.items():
            percentages[material] = count / total_pixels if total_pixels > 0 else 0
        
        # Create visualization
        visualization = self._create_visualization(image, material_masks, analysis_mask)
        
        # Determine dominant material
        dominant = max(percentages.items(), key=lambda x: x[1])
        
        return {
            'percentages': percentages,
            'pixel_counts': material_counts,
            'total_pixels': int(total_pixels),
            'dominant_material': dominant[0],
            'dominant_percentage': dominant[1],
            'visualization': visualization,
            'material_masks': material_masks,
        }
    
    def _create_material_mask(self, hsv: np.ndarray, material: str) -> np.ndarray:
        """Create binary mask for a specific material."""
        h, w = hsv.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        ranges = self.MATERIAL_COLORS[material]['ranges']
        for lower, upper in ranges:
            range_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            mask = cv2.bitwise_or(mask, range_mask)
        
        # Clean up with morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def _create_visualization(
        self,
        image: np.ndarray,
        material_masks: Dict[str, np.ndarray],
        analysis_mask: np.ndarray,
    ) -> np.ndarray:
        """Create color-coded material visualization with legend."""
        h, w = image.shape[:2]
        vis = image.copy()
        
        # Create overlay
        overlay = np.zeros_like(image)
        
        for material, mask in material_masks.items():
            if material in self.MATERIAL_COLORS:
                color = self.MATERIAL_COLORS[material]['color_bgr']
                overlay[mask > 0] = color
        
        # Mixed/unclassified gets neutral color
        all_classified = np.zeros((h, w), dtype=np.uint8)
        for mask in material_masks.values():
            all_classified = cv2.bitwise_or(all_classified, mask)
        
        unclassified = (analysis_mask > 0) & (all_classified == 0)
        overlay[unclassified] = (128, 128, 128)  # Gray for mixed
        
        # Blend - stronger overlay to see materials clearly
        vis = cv2.addWeighted(vis, 0.4, overlay, 0.6, 0)
        
        # Darken areas outside analysis mask significantly
        vis[analysis_mask == 0] = (vis[analysis_mask == 0] * 0.2).astype(np.uint8)
        
        # Add contour around analysis region
        contours, _ = cv2.findContours(analysis_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (255, 255, 255), 2)
        
        # Add legend
        legend_y = 30
        for material in ['concrete', 'wood', 'metal', 'mixed']:
            if material == 'mixed':
                color = (128, 128, 128)
            elif material in self.MATERIAL_COLORS:
                color = self.MATERIAL_COLORS[material]['color_bgr']
            else:
                continue
            cv2.rectangle(vis, (10, legend_y - 15), (30, legend_y + 5), color, -1)
            cv2.rectangle(vis, (10, legend_y - 15), (30, legend_y + 5), (255, 255, 255), 1)
            cv2.putText(vis, material.capitalize(), (40, legend_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            legend_y += 30
        
        return vis
    
    def create_spatial_material_map(
        self,
        image: np.ndarray,
        mask: np.ndarray = None,
        percentages: Dict = None,
    ) -> np.ndarray:
        """
        Create a clear spatial map showing material regions with embedded legend and percentages.
        Only analyzes within the mask (piles/containers).
        """
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        if mask is None:
            mask = np.ones((h, w), dtype=np.uint8) * 255
        
        analysis_mask = (mask > 0).astype(np.uint8)
        
        # Create output - start with darkened original
        output = (image * 0.25).astype(np.uint8)
        
        # VIBRANT, DISTINCT colors (BGR format)
        colors = {
            'concrete': (220, 220, 220),   # Bright white-gray
            'wood': (0, 140, 255),          # Bright orange
            'metal': (255, 100, 100),       # Bright blue
            'mixed': (128, 0, 128),         # Purple
        }
        
        # Calculate percentages if not provided
        if percentages is None:
            result = self.classify(image, mask)
            percentages = result['percentages']
        
        # Classify and color each pixel within mask
        material_counts = {}
        total_pile_pixels = analysis_mask.sum()
        
        for material in ['concrete', 'wood', 'metal']:
            mat_mask = self._create_material_mask(hsv, material)
            mat_mask = cv2.bitwise_and(mat_mask, mat_mask, mask=analysis_mask)
            material_counts[material] = (mat_mask > 0).sum()
            output[mat_mask > 0] = colors[material]
        
        # Unclassified within mask = mixed
        all_mats = np.zeros((h, w), dtype=np.uint8)
        for material in ['concrete', 'wood', 'metal']:
            mat_mask = self._create_material_mask(hsv, material)
            mat_mask = cv2.bitwise_and(mat_mask, mat_mask, mask=analysis_mask)
            all_mats = cv2.bitwise_or(all_mats, mat_mask)
        
        mixed_mask = (analysis_mask > 0) & (all_mats == 0)
        output[mixed_mask] = colors['mixed']
        material_counts['mixed'] = mixed_mask.sum()
        
        # Draw pile boundaries - thick bright green
        contours, _ = cv2.findContours(analysis_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, (0, 255, 0), 4)
        
        # === EMBEDDED LEGEND WITH PERCENTAGES ===
        # Calculate real percentages from this analysis
        pct = {}
        for mat, count in material_counts.items():
            pct[mat] = (count / total_pile_pixels * 100) if total_pile_pixels > 0 else 0
        
        # Legend panel - bottom left, large and readable
        panel_w, panel_h = 320, 200
        panel_x, panel_y = 20, h - panel_h - 20
        
        # Semi-transparent background
        overlay = output.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (20, 20, 20), -1)
        output = cv2.addWeighted(overlay, 0.85, output, 0.15, 0)
        cv2.rectangle(output, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (255, 255, 255), 2)
        
        # Title
        cv2.putText(output, "MATERIAL BREAKDOWN", (panel_x + 15, panel_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Legend entries with percentages
        y_offset = panel_y + 60
        for material, color in [('concrete', colors['concrete']), ('wood', colors['wood']), 
                                 ('metal', colors['metal']), ('mixed', colors['mixed'])]:
            # Color swatch
            cv2.rectangle(output, (panel_x + 15, y_offset - 15), (panel_x + 45, y_offset + 10), color, -1)
            cv2.rectangle(output, (panel_x + 15, y_offset - 15), (panel_x + 45, y_offset + 10), (255, 255, 255), 1)
            
            # Material name and percentage
            pct_val = pct.get(material, 0)
            text = f"{material.upper()}: {pct_val:.1f}%"
            cv2.putText(output, text, (panel_x + 55, y_offset + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            
            # Progress bar
            bar_x = panel_x + 200
            bar_w = 100
            cv2.rectangle(output, (bar_x, y_offset - 10), (bar_x + bar_w, y_offset + 5), (60, 60, 60), -1)
            filled_w = int(bar_w * pct_val / 100)
            if filled_w > 0:
                cv2.rectangle(output, (bar_x, y_offset - 10), (bar_x + filled_w, y_offset + 5), color, -1)
            cv2.rectangle(output, (bar_x, y_offset - 10), (bar_x + bar_w, y_offset + 5), (255, 255, 255), 1)
            
            y_offset += 35
        
        return output
    
    def _empty_result(self) -> Dict:
        """Return empty result when no pixels to analyze."""
        return {
            'percentages': {m: 0.0 for m in self.materials + ['mixed']},
            'pixel_counts': {m: 0 for m in self.materials + ['mixed']},
            'total_pixels': 0,
            'dominant_material': 'unknown',
            'dominant_percentage': 0.0,
            'visualization': None,
            'material_masks': {},
        }
    
    def analyze_histogram(
        self,
        image: np.ndarray,
        mask: np.ndarray = None,
    ) -> Dict:
        """
        Analyze color histogram within masked region.
        
        Args:
            image: BGR image.
            mask: Binary mask.
        
        Returns:
            Dict with histogram data.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate histograms
        hist_h = cv2.calcHist([hsv], [0], mask, [180], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], mask, [256], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], mask, [256], [0, 256])
        
        # Normalize
        hist_h = hist_h.flatten() / (hist_h.sum() + 1e-8)
        hist_s = hist_s.flatten() / (hist_s.sum() + 1e-8)
        hist_v = hist_v.flatten() / (hist_v.sum() + 1e-8)
        
        return {
            'hue': hist_h,
            'saturation': hist_s,
            'value': hist_v,
            'mean_hue': float(np.average(np.arange(180), weights=hist_h)),
            'mean_saturation': float(np.average(np.arange(256), weights=hist_s)),
            'mean_value': float(np.average(np.arange(256), weights=hist_v)),
        }


def classify_materials(
    image: np.ndarray,
    mask: np.ndarray = None,
) -> Dict:
    """
    Convenience function for material classification.
    
    Args:
        image: BGR image.
        mask: Optional binary mask.
    
    Returns:
        Classification results.
    """
    classifier = MaterialClassifier()
    return classifier.classify(image, mask)


def get_material_summary(result: Dict) -> str:
    """Get human-readable summary of material classification."""
    pct = result['percentages']
    lines = [
        f"Material Breakdown:",
        f"  Concrete: {pct.get('concrete', 0)*100:.1f}%",
        f"  Wood: {pct.get('wood', 0)*100:.1f}%",
        f"  Metal: {pct.get('metal', 0)*100:.1f}%",
        f"  Mixed/Other: {pct.get('mixed', 0)*100:.1f}%",
        f"  Dominant: {result['dominant_material']} ({result['dominant_percentage']*100:.1f}%)",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    from .scraper import list_available_images, DATA_DIR
    
    images = list_available_images()
    if images:
        print(f"[*] Testing material classification on: {images[0]}")
        
        classifier = MaterialClassifier()
        image = cv2.imread(str(images[0]))
        result = classifier.classify(image)
        
        print(get_material_summary(result))
        
        # Save visualization
        if result['visualization'] is not None:
            output_path = DATA_DIR / "materials_test.jpg"
            cv2.imwrite(str(output_path), result['visualization'])
            print(f"\n[+] Saved visualization to: {output_path}")
    else:
        print("[!] No images available. Run scraper.py first.")
