"""
Revenue estimation module.
Estimates material value based on containers, pallets, volumes, and material classification.
"""

from datetime import datetime
from typing import Dict, Optional
from pathlib import Path

from .timeseries_db import TimeseriesDB


# Default material prices (per ton, USD)
DEFAULT_MATERIAL_PRICES = {
    'concrete': 20,      # Recycled concrete aggregate
    'wood': 50,          # Recycled wood/chips
    'metal': 300,        # Mixed scrap metal
    'copper': 8000,      # Copper scrap
    'aluminum': 2000,    # Aluminum scrap
    'steel': 300,        # Steel scrap
    'mixed': 100,        # Mixed C&D debris
}


class RevenueEstimator:
    """Estimates revenue potential from visible materials."""
    
    def __init__(
        self,
        db_path: str = None,
        material_prices: Dict = None,
    ):
        """
        Initialize revenue estimator.
        
        Args:
            db_path: Path to database
            material_prices: Dict of material prices per ton (USD)
        """
        self.db = TimeseriesDB(db_path)
        self.material_prices = material_prices or DEFAULT_MATERIAL_PRICES.copy()
    
    def estimate_revenue(
        self,
        date: str,
        material_prices: Dict = None,
    ) -> Dict:
        """
        Estimate revenue from materials visible on a specific date.
        
        Args:
            date: Date (ISO format)
            material_prices: Override material prices
        
        Returns:
            Revenue estimation dict
        """
        images = self.db.get_all_images()
        
        # Find image for this date
        target_image = None
        for img in images:
            if img['date'].startswith(date[:10]):  # Match date part
                target_image = img
                break
        
        if not target_image:
            return {
                'date': date,
                'estimated_value': 0,
                'confidence': 'low',
                'breakdown': {},
                'error': 'No image found for date',
            }
        
        # Get containers and pallets
        containers = self.db.get_containers_by_date_range(date[:10], date[:10])
        container_count = len([c for c in containers if c['image_id'] == target_image['id']])
        
        # Get material breakdown
        import json
        materials = {}
        if target_image.get('materials_json'):
            materials = json.loads(target_image['materials_json'])
        
        tonnage = target_image.get('tonnage_estimate', 0) or 0
        
        # Estimate revenue
        prices = material_prices or self.material_prices
        
        breakdown = {}
        total_value = 0
        
        for material, percentage in materials.items():
            material_tonnage = tonnage * (percentage / 100)
            
            # Map material to price
            price_key = material.lower()
            if price_key not in prices:
                # Try to infer: 'wood' -> 'wood', 'metal' -> 'metal', etc.
                if 'copper' in price_key:
                    price_key = 'copper'
                elif 'aluminum' in price_key or 'aluminium' in price_key:
                    price_key = 'aluminum'
                elif 'steel' in price_key:
                    price_key = 'steel'
                else:
                    price_key = 'mixed'
            
            price_per_ton = prices.get(price_key, prices['mixed'])
            material_value = material_tonnage * price_per_ton
            
            breakdown[material] = {
                'tonnage': round(material_tonnage, 1),
                'price_per_ton': price_per_ton,
                'value': round(material_value, 0),
                'percentage': percentage,
            }
            
            total_value += material_value
        
        # Add container-based estimate (if containers detected)
        if container_count > 0:
            # Estimate: each container ~20-25 tons capacity
            container_tonnage = container_count * 22.5
            container_value = container_tonnage * prices.get('mixed', 100)
            
            breakdown['containers'] = {
                'count': container_count,
                'estimated_tonnage': container_tonnage,
                'value': round(container_value, 0),
            }
            
            total_value += container_value
        
        # Confidence based on data quality
        confidence = 'medium'
        if tonnage == 0:
            confidence = 'low'
        elif container_count > 0 and len(materials) > 0:
            confidence = 'high'
        
        return {
            'date': date,
            'estimated_value': round(total_value, 0),
            'confidence': confidence,
            'breakdown': breakdown,
            'tonnage': round(tonnage, 1),
            'container_count': container_count,
        }
    
    def estimate_revenue_range(
        self,
        start_date: str,
        end_date: str,
        material_prices: Dict = None,
    ) -> Dict:
        """
        Estimate total revenue over a date range.
        
        Args:
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            material_prices: Override material prices
        
        Returns:
            Revenue estimation for date range
        """
        images = self.db.get_all_images()
        
        # Filter by date range
        images_in_range = [
            img for img in images
            if start_date <= img['date'] <= end_date
        ]
        
        if not images_in_range:
            return {
                'start_date': start_date,
                'end_date': end_date,
                'total_value': 0,
                'average_per_image': 0,
                'breakdown': {},
            }
        
        total_value = 0
        all_breakdowns = []
        
        for img in images_in_range:
            estimate = self.estimate_revenue(img['date'], material_prices)
            total_value += estimate['estimated_value']
            all_breakdowns.append(estimate['breakdown'])
        
        # Aggregate breakdown
        aggregated = {}
        for breakdown in all_breakdowns:
            for material, data in breakdown.items():
                if material not in aggregated:
                    aggregated[material] = {'value': 0, 'tonnage': 0}
                aggregated[material]['value'] += data.get('value', 0)
                aggregated[material]['tonnage'] += data.get('tonnage', 0)
        
        return {
            'start_date': start_date,
            'end_date': end_date,
            'total_value': round(total_value, 0),
            'average_per_image': round(total_value / len(images_in_range), 0),
            'image_count': len(images_in_range),
            'breakdown': aggregated,
        }


if __name__ == "__main__":
    # Test revenue estimation
    estimator = RevenueEstimator()
    
    print("[*] Testing revenue estimation...")
    
    # Test single date
    estimate = estimator.estimate_revenue("2021-12-01")
    print(f"\n[+] Revenue estimate for 2021-12-01:")
    print(f"    Total value: ${estimate['estimated_value']:,.0f}")
    print(f"    Confidence: {estimate['confidence']}")
    print(f"    Tonnage: {estimate['tonnage']:.1f} tons")
    
    if estimate['breakdown']:
        print("    Breakdown:")
        for material, data in estimate['breakdown'].items():
            if isinstance(data, dict) and 'value' in data:
                print(f"      {material}: ${data['value']:,.0f}")
    
    # Test date range
    range_estimate = estimator.estimate_revenue_range("2021-01-01", "2022-12-31")
    print(f"\n[+] Revenue estimate for 2021-2022:")
    print(f"    Total value: ${range_estimate['total_value']:,.0f}")
    print(f"    Average per image: ${range_estimate['average_per_image']:,.0f}")
