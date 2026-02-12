"""
Batch processor for time-lapse images.
Processes all dated images through the pipeline and stores results.
"""

from pathlib import Path
from typing import List, Dict, Optional
import cv2
from tqdm import tqdm

from .timelapse import find_timelapse_images, extract_date_from_filename
from .pipeline import DebrisAnalysisPipeline
from .container_detector import ContainerDetector
from .timeseries_db import TimeseriesDB
from .height_estimator import CameraHeightEstimator


class BatchProcessor:
    """Processes time-lapse images in batch."""
    
    def __init__(
        self,
        use_sam: bool = True,
        auto_height: bool = True,
        db_path: Path = None,
    ):
        """
        Initialize batch processor.
        
        Args:
            use_sam: Whether to use SAM for segmentation.
            auto_height: Whether to auto-estimate camera height.
            db_path: Path to database. Defaults to data/timeseries/timelapse.db
        """
        self.use_sam = use_sam
        self.auto_height = auto_height
        self.db = TimeseriesDB(db_path)
        self.height_estimator = CameraHeightEstimator()
    
    def process_all(
        self,
        data_dir: Path = None,
        overwrite: bool = False,
    ) -> Dict:
        """
        Process all timelapse images.
        
        Args:
            data_dir: Directory containing timelapse images.
            overwrite: Whether to reprocess existing images.
        
        Returns:
            Summary of processing results
        """
        images = find_timelapse_images(data_dir)
        
        if not images:
            return {'processed': 0, 'skipped': 0, 'errors': []}
        
        processed = 0
        skipped = 0
        errors = []
        
        print(f"[*] Processing {len(images)} timelapse images...")
        
        for img_path, date in tqdm(images, desc="Processing"):
            try:
                # Check if already processed
                if not overwrite:
                    existing = self.db.get_all_images()
                    if any(img['filename'] == img_path.name for img in existing):
                        skipped += 1
                        continue
                
                # Process image
                result = self._process_image(img_path, date)
                
                if result:
                    processed += 1
                else:
                    errors.append(f"{img_path.name}: Processing returned None")
                    
            except Exception as e:
                errors.append(f"{img_path.name}: {str(e)}")
                print(f"[!] Error processing {img_path.name}: {e}")
        
        return {
            'processed': processed,
            'skipped': skipped,
            'errors': errors,
            'total': len(images),
        }
    
    def _process_image(
        self,
        img_path: Path,
        date,
    ) -> Optional[Dict]:
        """
        Process a single image.
        
        Args:
            img_path: Path to image file
            date: datetime object
        
        Returns:
            Processing result dict or None on error
        """
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            return None
        
        h, w = image.shape[:2]
        
        # Estimate or use default camera height
        if self.auto_height:
            height_result = self.height_estimator.estimate(image)
            camera_height = height_result['estimated_height_m']
        else:
            camera_height = 30.0
        
        # Initialize pipeline
        pipeline = DebrisAnalysisPipeline(
            use_sam=self.use_sam,
            depth_model='small',
            camera_height_m=camera_height,
        )
        
        # Run analysis
        analysis_result = pipeline.analyze(image, timestamp=date)
        
        # Detect containers and pallets
        container_detector = ContainerDetector(use_sam=self.use_sam)
        container_result = container_detector.detect(image, camera_height_m=camera_height)
        
        # Store in database
        image_id = self.db.add_image(
            filename=img_path.name,
            filepath=str(img_path),
            date=date,
            camera_height_m=camera_height,
            image_width=w,
            image_height=h,
        )
        
        # Store analysis results
        if 'volume' in analysis_result:
            self.db.add_analysis_result(
                image_id=image_id,
                volume_m3=analysis_result['volume']['volume_m3'],
                pile_area_m2=analysis_result['volume']['pile_area_m2'],
                avg_height_m=analysis_result['volume']['avg_height_m'],
                max_height_m=analysis_result['volume']['max_height_m'],
                tonnage_estimate=analysis_result['volume']['tonnage_estimate'],
                materials=analysis_result['materials']['percentages'],
                dominant_material=analysis_result['materials']['dominant'],
            )
        
        # Store containers and pallets
        if container_result['containers']:
            self.db.add_containers(image_id, container_result['containers'])
        
        if container_result['pallets']:
            self.db.add_pallets(image_id, container_result['pallets'])
        
        return {
            'image_id': image_id,
            'date': date.isoformat(),
            'volume_m3': analysis_result.get('volume', {}).get('volume_m3', 0),
            'containers': container_result['container_count'],
            'pallets': container_result['pallet_count'],
        }
    
    def get_processing_summary(self) -> Dict:
        """Get summary of processed data."""
        return self.db.get_summary_stats()


if __name__ == "__main__":
    # Test batch processing
    processor = BatchProcessor(use_sam=False, auto_height=True)
    
    print("[*] Starting batch processing...")
    result = processor.process_all()
    
    print(f"\n[+] Processing complete:")
    print(f"    Processed: {result['processed']}")
    print(f"    Skipped: {result['skipped']}")
    print(f"    Errors: {len(result['errors'])}")
    
    if result['errors']:
        print("\n[!] Errors:")
        for error in result['errors']:
            print(f"    - {error}")
    
    # Show summary
    summary = processor.get_processing_summary()
    print(f"\n[+] Database summary:")
    print(f"    Total images: {summary['total_images']}")
    print(f"    Total containers: {summary['total_containers']}")
    print(f"    Total pallets: {summary['total_pallets']}")
    if summary['date_range']:
        print(f"    Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
