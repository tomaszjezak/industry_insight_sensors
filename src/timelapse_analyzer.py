"""
Time-lapse change analyzer.
Compares consecutive images to detect arrivals, departures, and changes.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from .timeseries_db import TimeseriesDB
from .change_detection import ChangeDetector


class TimelapseAnalyzer:
    """Analyzes changes over time in timelapse images."""
    
    def __init__(self, db_path: str = None):
        """
        Initialize analyzer.
        
        Args:
            db_path: Path to database
        """
        self.db = TimeseriesDB(db_path)
        self.change_detector = ChangeDetector(method='combined')
    
    def analyze_changes(
        self,
        start_date: str = None,
        end_date: str = None,
    ) -> Dict:
        """
        Analyze changes between consecutive images in date range.
        
        Args:
            start_date: Start date (ISO format). If None, uses earliest.
            end_date: End date (ISO format). If None, uses latest.
        
        Returns:
            Dict with change analysis results
        """
        images = self.db.get_all_images()
        
        if len(images) < 2:
            return {
                'changes': [],
                'summary': {
                    'total_changes': 0,
                    'arrivals': 0,
                    'departures': 0,
                }
            }
        
        # Filter by date range
        if start_date:
            images = [img for img in images if img['date'] >= start_date]
        if end_date:
            images = [img for img in images if img['date'] <= end_date]
        
        images.sort(key=lambda x: x['date'])
        
        changes = []
        
        for i in range(len(images) - 1):
            img1 = images[i]
            img2 = images[i + 1]
            
            change = self._compare_images(img1, img2)
            if change:
                changes.append(change)
        
        # Calculate summary
        total_arrivals = sum(c.get('container_arrivals', 0) + c.get('pallet_arrivals', 0) for c in changes)
        total_departures = sum(c.get('container_departures', 0) + c.get('pallet_departures', 0) for c in changes)
        
        return {
            'changes': changes,
            'summary': {
                'total_changes': len(changes),
                'arrivals': total_arrivals,
                'departures': total_departures,
                'net_change': total_arrivals - total_departures,
            }
        }
    
    def _compare_images(self, img1: Dict, img2: Dict) -> Optional[Dict]:
        """
        Compare two images and detect changes.
        
        Args:
            img1: First image record from database
            img2: Second image record from database
        
        Returns:
            Change analysis dict or None
        """
        from pathlib import Path
        
        # Load images
        img1_path = Path(img1['filepath'])
        img2_path = Path(img2['filepath'])
        
        if not img1_path.exists() or not img2_path.exists():
            return None
        
        image1 = cv2.imread(str(img1_path))
        image2 = cv2.imread(str(img2_path))
        
        if image1 is None or image2 is None:
            return None
        
        # Get containers and pallets for each image
        containers1 = self.db.get_containers_by_date_range(img1['date'], img1['date'])
        containers2 = self.db.get_containers_by_date_range(img2['date'], img2['date'])
        
        # Simple change detection: count differences
        container_count1 = len([c for c in containers1 if c['image_id'] == img1['id']])
        container_count2 = len([c for c in containers2 if c['image_id'] == img2['id']])
        
        # Get pallet counts (simplified - would need pallet query method)
        # For now, estimate from analysis results
        volume_change = (img2.get('volume_m3', 0) or 0) - (img1.get('volume_m3', 0) or 0)
        area_change = (img2.get('pile_area_m2', 0) or 0) - (img1.get('pile_area_m2', 0) or 0)
        
        # Detect arrivals/departures
        container_arrivals = max(0, container_count2 - container_count1)
        container_departures = max(0, container_count1 - container_count2)
        
        # Estimate pallet changes from volume/area changes
        # Rough heuristic: significant volume increase = arrivals, decrease = departures
        pallet_arrivals = 0
        pallet_departures = 0
        if volume_change > 100:  # Significant increase
            pallet_arrivals = int(volume_change / 50)  # Rough estimate
        elif volume_change < -100:  # Significant decrease
            pallet_departures = int(abs(volume_change) / 50)
        
        return {
            'date_from': img1['date'],
            'date_to': img2['date'],
            'days_apart': (datetime.fromisoformat(img2['date']) - datetime.fromisoformat(img1['date'])).days,
            'container_arrivals': container_arrivals,
            'container_departures': container_departures,
            'container_net_change': container_count2 - container_count1,
            'pallet_arrivals': pallet_arrivals,
            'pallet_departures': pallet_departures,
            'volume_change_m3': round(volume_change, 2),
            'area_change_m2': round(area_change, 2),
            'tonnage_change': round((img2.get('tonnage_estimate', 0) or 0) - (img1.get('tonnage_estimate', 0) or 0), 1),
        }
    
    def get_inventory_timeline(
        self,
        start_date: str = None,
        end_date: str = None,
    ) -> List[Dict]:
        """
        Get inventory levels over time.
        
        Args:
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
        
        Returns:
            List of inventory snapshots
        """
        images = self.db.get_all_images()
        
        if start_date:
            images = [img for img in images if img['date'] >= start_date]
        if end_date:
            images = [img for img in images if img['date'] <= end_date]
        
        images.sort(key=lambda x: x['date'])
        
        timeline = []
        for img in images:
            containers = self.db.get_containers_by_date_range(img['date'], img['date'])
            container_count = len([c for c in containers if c['image_id'] == img['id']])
            
            timeline.append({
                'date': img['date'],
                'filename': img['filename'],
                'volume_m3': img.get('volume_m3', 0) or 0,
                'area_m2': img.get('pile_area_m2', 0) or 0,
                'tonnage': img.get('tonnage_estimate', 0) or 0,
                'containers': container_count,
                'dominant_material': img.get('dominant_material', 'unknown'),
            })
        
        return timeline


if __name__ == "__main__":
    # Test change analysis
    analyzer = TimelapseAnalyzer()
    
    print("[*] Analyzing changes over time...")
    changes = analyzer.analyze_changes()
    
    print(f"\n[+] Found {changes['summary']['total_changes']} change periods")
    print(f"    Arrivals: {changes['summary']['arrivals']}")
    print(f"    Departures: {changes['summary']['departures']}")
    print(f"    Net change: {changes['summary']['net_change']}")
    
    print("\n[+] Change details:")
    for change in changes['changes'][:5]:
        print(f"    {change['date_from']} → {change['date_to']}: "
              f"{change['container_arrivals']} arrivals, "
              f"{change['container_departures']} departures, "
              f"{change['volume_change_m3']:+.1f}m³ volume change")
