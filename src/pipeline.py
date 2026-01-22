"""
Main pipeline orchestrator for C&D debris analysis.
Combines segmentation, depth estimation, volume calculation, 
material classification, and change detection into a unified flow.
"""

import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
import cv2
import json

from .segmentation import PileSegmenter
from .depth import DepthEstimator
from .volume import VolumeCalculator
from .materials import MaterialClassifier
from .change_detection import ChangeDetector
from .height_estimator import CameraHeightEstimator


class DebrisAnalysisPipeline:
    """
    End-to-end pipeline for analyzing C&D debris piles.
    
    Typical workflow:
    1. Load image
    2. Segment pile from background
    3. Estimate depth
    4. Calculate volume
    5. Classify materials
    6. (Optional) Compare with previous image for change detection
    """
    
    def __init__(
        self,
        use_sam: bool = True,
        depth_model: str = 'small',
        camera_height_m: float = None,  # None = auto-estimate
        device: str = None,
    ):
        """
        Initialize pipeline components.
        
        Args:
            use_sam: Whether to use SAM for segmentation (requires model).
            depth_model: MiDaS model type - 'small', 'hybrid', or 'large'.
            camera_height_m: Camera height for volume calculation. None = auto-estimate.
            device: 'cuda' or 'cpu'. Auto-detected if None.
        """
        self.camera_height_m = camera_height_m  # Can be None for auto
        self.auto_height = camera_height_m is None
        
        print("[*] Initializing C&D Debris Analysis Pipeline...")
        
        # Initialize components
        self.segmenter = PileSegmenter(use_sam=use_sam, device=device)
        self.depth_estimator = DepthEstimator(model_type=depth_model, device=device)
        self.volume_calculator = None  # Created per-image if auto-height
        self.material_classifier = MaterialClassifier()
        self.change_detector = ChangeDetector()
        self.height_estimator = CameraHeightEstimator()
        
        if not self.auto_height:
            self.volume_calculator = VolumeCalculator(camera_height_m=camera_height_m)
        
        print("[+] Pipeline initialized")
    
    def analyze(
        self,
        image: np.ndarray,
        previous_result: Dict = None,
        timestamp: datetime = None,
    ) -> Dict:
        """
        Run full analysis on an image.
        
        Args:
            image: BGR image to analyze.
            previous_result: Optional previous analysis result for change detection.
            timestamp: Timestamp for this analysis. Defaults to now.
        
        Returns:
            Dict with all analysis results.
        """
        timestamp = timestamp or datetime.now()
        h, w = image.shape[:2]
        
        # Auto-estimate camera height if not set
        if self.auto_height:
            print("[0/5] Estimating camera height...")
            height_result = self.height_estimator.estimate(image)
            camera_height = height_result['estimated_height_m']
            print(f"      Estimated: {camera_height}m ({height_result['recommendation']})")
            self.volume_calculator = VolumeCalculator(camera_height_m=camera_height)
        else:
            camera_height = self.camera_height_m
            if self.volume_calculator is None:
                self.volume_calculator = VolumeCalculator(camera_height_m=camera_height)
        
        result = {
            'timestamp': timestamp.isoformat(),
            'image_size': {'width': w, 'height': h},
            'camera_height_m': camera_height,
            'height_auto_estimated': self.auto_height,
        }
        
        if self.auto_height:
            result['height_estimation'] = {
                'estimated': height_result['estimated_height_m'],
                'confidence': height_result['confidence'],
                'category': height_result['recommendation'],
            }
        
        # Step 1: Segmentation
        print("[1/5] Segmenting pile...")
        seg_result = self.segmenter.segment(image)
        result['segmentation'] = {
            'method': seg_result['method'],
            'num_segments': len(seg_result['segments']),
            'segments': seg_result['segments'],
        }
        mask = seg_result['mask']
        
        # Store mask for change detection
        result['_mask'] = mask
        result['_image'] = image
        
        # Step 2: Depth estimation
        print("[2/5] Estimating depth...")
        depth_result = self.depth_estimator.estimate(image)
        result['depth'] = {
            'method': depth_result['method'],
        }
        
        # Step 3: Volume calculation
        print("[3/5] Calculating volume...")
        volume_result = self.volume_calculator.calculate(
            depth_result['depth_normalized'],
            mask
        )
        tonnage = self.volume_calculator.estimate_tonnage(
            volume_result['volume_m3'],
            'mixed'  # Default to mixed; will update based on material classification
        )
        
        result['volume'] = {
            'volume_m3': round(volume_result['volume_m3'], 2),
            'pile_area_m2': round(volume_result['pile_area_m2'], 2),
            'avg_height_m': round(volume_result['avg_height_m'], 2),
            'max_height_m': round(volume_result['max_height_m'], 2),
            'tonnage_estimate': round(tonnage['tonnage'], 1),
            'tonnage_range': {
                'low': round(tonnage['tonnage_range']['low'], 1),
                'high': round(tonnage['tonnage_range']['high'], 1),
            },
        }
        
        # Step 4: Material classification
        print("[4/5] Classifying materials...")
        material_result = self.material_classifier.classify(image, mask)
        
        # Round percentages
        materials_pct = {
            k: round(v * 100, 1) 
            for k, v in material_result['percentages'].items()
        }
        
        result['materials'] = {
            'percentages': materials_pct,
            'dominant': material_result['dominant_material'],
            'dominant_percentage': round(material_result['dominant_percentage'] * 100, 1),
        }
        
        # Update tonnage estimate based on dominant material
        if material_result['dominant_material'] != 'mixed':
            updated_tonnage = self.volume_calculator.estimate_tonnage(
                volume_result['volume_m3'],
                material_result['dominant_material']
            )
            result['volume']['tonnage_by_dominant'] = round(updated_tonnage['tonnage'], 1)
        
        # Step 5: Change detection (if previous result available)
        if previous_result is not None and '_image' in previous_result:
            print("[5/5] Detecting changes...")
            prev_image = previous_result['_image']
            prev_mask = previous_result.get('_mask')
            
            change_result = self.change_detector.detect(
                prev_image, image,
                prev_mask, mask
            )
            
            result['changes'] = {
                'change_type': change_result['change_type'],
                'change_percentage': round(change_result['change_percentage'] * 100, 1),
                'pile_net_change_pct': round(
                    change_result.get('mask_net_change_pct', 0) * 100, 1
                ),
                'ssim_score': round(change_result.get('ssim_score', 0), 3),
            }
            result['_change_vis'] = change_result['visualization']
        else:
            print("[5/5] No previous image for change detection")
            result['changes'] = None
        
        # Store visualizations (not serializable to JSON, for dashboard use)
        result['_visualizations'] = {
            'segmentation': self.segmenter.visualize(image, seg_result),
            'depth': depth_result['depth_colored'],
            'height': self.volume_calculator.create_height_visualization(
                depth_result['depth_normalized'], mask
            ),
            'materials': material_result['visualization'],
            'materials_spatial': self.material_classifier.create_spatial_material_map(
                image, mask, percentages=material_result['percentages']
            ),
        }
        
        print("[+] Analysis complete")
        return result
    
    def analyze_image_file(
        self,
        image_path: str | Path,
        previous_result: Dict = None,
    ) -> Dict:
        """
        Analyze an image from file path.
        
        Args:
            image_path: Path to image file.
            previous_result: Optional previous result for change detection.
        
        Returns:
            Analysis result dict.
        """
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))
        
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Get timestamp from file or use current time
        try:
            mtime = datetime.fromtimestamp(image_path.stat().st_mtime)
        except:
            mtime = datetime.now()
        
        result = self.analyze(image, previous_result, timestamp=mtime)
        result['source_file'] = str(image_path.name)
        
        return result
    
    def get_summary(self, result: Dict) -> str:
        """Get human-readable summary of analysis."""
        lines = [
            "=" * 50,
            "C&D DEBRIS PILE ANALYSIS",
            "=" * 50,
            f"Timestamp: {result['timestamp']}",
            f"Image: {result.get('source_file', 'N/A')}",
            "",
            "VOLUME ESTIMATION:",
            f"  Volume: {result['volume']['volume_m3']} m³",
            f"  Area: {result['volume']['pile_area_m2']} m²",
            f"  Avg Height: {result['volume']['avg_height_m']} m",
            f"  Max Height: {result['volume']['max_height_m']} m",
            f"  Est. Tonnage: {result['volume']['tonnage_estimate']} tons",
            f"    (range: {result['volume']['tonnage_range']['low']}-{result['volume']['tonnage_range']['high']} tons)",
            "",
            "MATERIAL BREAKDOWN:",
        ]
        
        for material, pct in result['materials']['percentages'].items():
            lines.append(f"  {material.capitalize()}: {pct}%")
        
        lines.append(f"  Dominant: {result['materials']['dominant']} ({result['materials']['dominant_percentage']}%)")
        
        if result['changes']:
            lines.extend([
                "",
                "CHANGES FROM PREVIOUS:",
                f"  Change Type: {result['changes']['change_type']}",
                f"  Changed Area: {result['changes']['change_percentage']}%",
                f"  Pile Size Change: {result['changes']['pile_net_change_pct']}%",
            ])
        
        lines.append("=" * 50)
        return "\n".join(lines)
    
    def to_json(self, result: Dict) -> str:
        """Convert result to JSON (excluding visualizations)."""
        # Create clean copy without numpy arrays and images
        clean = {}
        for k, v in result.items():
            if not k.startswith('_'):
                clean[k] = v
        
        return json.dumps(clean, indent=2)
    
    def save_result(self, result: Dict, output_dir: Path) -> Dict[str, Path]:
        """
        Save analysis result and visualizations to files.
        
        Args:
            result: Analysis result.
            output_dir: Directory to save files.
        
        Returns:
            Dict of saved file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Base name from timestamp
        ts = result['timestamp'].replace(':', '-').replace('.', '-')
        base_name = f"analysis_{ts}"
        
        saved = {}
        
        # Save JSON
        json_path = output_dir / f"{base_name}.json"
        with open(json_path, 'w') as f:
            f.write(self.to_json(result))
        saved['json'] = json_path
        
        # Save visualizations
        vis = result.get('_visualizations', {})
        for name, img in vis.items():
            if img is not None:
                img_path = output_dir / f"{base_name}_{name}.jpg"
                cv2.imwrite(str(img_path), img)
                saved[name] = img_path
        
        # Save change visualization if present
        if '_change_vis' in result:
            change_path = output_dir / f"{base_name}_changes.jpg"
            cv2.imwrite(str(change_path), result['_change_vis'])
            saved['changes'] = change_path
        
        return saved


def analyze_image(
    image_path: str | Path,
    camera_height_m: float = 3.0,
) -> Dict:
    """
    Convenience function for single image analysis.
    
    Args:
        image_path: Path to image file.
        camera_height_m: Assumed camera height.
    
    Returns:
        Analysis result dict.
    """
    pipeline = DebrisAnalysisPipeline(
        use_sam=True,
        depth_model='small',
        camera_height_m=camera_height_m
    )
    return pipeline.analyze_image_file(image_path)


if __name__ == "__main__":
    from .scraper import list_available_images, DATA_DIR
    
    images = list_available_images()
    if images:
        print(f"\n[*] Running full pipeline on: {images[0]}")
        
        # Initialize pipeline
        pipeline = DebrisAnalysisPipeline(
            use_sam=True,  # Will fall back to OpenCV if SAM not available
            depth_model='small',
            camera_height_m=3.0,
        )
        
        # Analyze first image
        result = pipeline.analyze_image_file(images[0])
        
        # Print summary
        print("\n" + pipeline.get_summary(result))
        
        # Save results
        output_dir = DATA_DIR / "results"
        saved = pipeline.save_result(result, output_dir)
        print(f"\n[+] Saved results to: {output_dir}")
        for name, path in saved.items():
            print(f"    - {name}: {path.name}")
        
        # If we have a second image, show change detection
        if len(images) >= 2:
            print(f"\n[*] Analyzing second image with change detection: {images[1]}")
            result2 = pipeline.analyze_image_file(images[1], previous_result=result)
            print("\n" + pipeline.get_summary(result2))
    else:
        print("[!] No images available. Run scraper.py first.")
