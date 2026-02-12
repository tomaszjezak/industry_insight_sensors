"""
Query engine for time-series data.
Supports structured queries on timelapse analysis results.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from .timeseries_db import TimeseriesDB
from .timelapse_analyzer import TimelapseAnalyzer
from .revenue_estimator import RevenueEstimator


class QueryEngine:
    """Query engine for timelapse data."""
    
    def __init__(self, db_path: str = None):
        """
        Initialize query engine.
        
        Args:
            db_path: Path to database
        """
        self.db = TimeseriesDB(db_path)
        self.analyzer = TimelapseAnalyzer(db_path)
        self.estimator = RevenueEstimator(db_path)
    
    def query(
        self,
        query_type: str,
        **kwargs
    ) -> Dict:
        """
        Execute a structured query.
        
        Args:
            query_type: Type of query ('inventory', 'containers', 'changes', 'revenue', 'timeline')
            **kwargs: Query-specific parameters
        
        Returns:
            Query results
        """
        if query_type == 'inventory':
            return self._query_inventory(**kwargs)
        elif query_type == 'containers':
            return self._query_containers(**kwargs)
        elif query_type == 'changes':
            return self._query_changes(**kwargs)
        elif query_type == 'revenue':
            return self._query_revenue(**kwargs)
        elif query_type == 'timeline':
            return self._query_timeline(**kwargs)
        else:
            return {'error': f'Unknown query type: {query_type}'}
    
    def _query_inventory(
        self,
        date: str = None,
        date_range: Tuple[str, str] = None,
    ) -> Dict:
        """
        Query inventory levels.
        
        Args:
            date: Single date (ISO format)
            date_range: (start_date, end_date) tuple
        
        Returns:
            Inventory data
        """
        if date:
            images = self.db.get_all_images()
            target = [img for img in images if img['date'].startswith(date[:10])]
            if target:
                img = target[0]
                containers = self.db.get_containers_by_date_range(date[:10], date[:10])
                container_count = len([c for c in containers if c['image_id'] == img['id']])
                
                return {
                    'date': date,
                    'volume_m3': img.get('volume_m3', 0) or 0,
                    'area_m2': img.get('pile_area_m2', 0) or 0,
                    'tonnage': img.get('tonnage_estimate', 0) or 0,
                    'containers': container_count,
                    'dominant_material': img.get('dominant_material', 'unknown'),
                }
            return {'error': f'No data for date: {date}'}
        
        elif date_range:
            start, end = date_range
            timeline = self.analyzer.get_inventory_timeline(start, end)
            return {
                'date_range': {'start': start, 'end': end},
                'snapshots': timeline,
                'count': len(timeline),
            }
        
        else:
            # Return all
            timeline = self.analyzer.get_inventory_timeline()
            return {
                'snapshots': timeline,
                'count': len(timeline),
            }
    
    def _query_containers(
        self,
        date_range: Tuple[str, str] = None,
    ) -> Dict:
        """
        Query container counts and changes.
        
        Args:
            date_range: (start_date, end_date) tuple
        
        Returns:
            Container statistics
        """
        if date_range:
            start, end = date_range
            containers = self.db.get_containers_by_date_range(start, end)
            
            # Group by date
            by_date = {}
            for c in containers:
                date = c['date'][:10]
                if date not in by_date:
                    by_date[date] = []
                by_date[date].append(c)
            
            # Calculate changes
            dates = sorted(by_date.keys())
            arrivals = 0
            departures = 0
            
            if len(dates) >= 2:
                count1 = len(by_date[dates[0]])
                count2 = len(by_date[dates[-1]])
                arrivals = max(0, count2 - count1)
                departures = max(0, count1 - count2)
            
            return {
                'date_range': {'start': start, 'end': end},
                'total_containers': len(containers),
                'containers_by_date': {date: len(containers) for date, containers in by_date.items()},
                'arrivals': arrivals,
                'departures': departures,
                'net_change': arrivals - departures,
            }
        
        else:
            containers = self.db.get_containers_by_date_range('2015-01-01', '2030-01-01')
            return {
                'total_containers': len(containers),
                'containers': containers[:10],  # First 10
            }
    
    def _query_changes(
        self,
        date_range: Tuple[str, str] = None,
    ) -> Dict:
        """
        Query changes between dates.
        
        Args:
            date_range: (start_date, end_date) tuple
        
        Returns:
            Change analysis
        """
        if date_range:
            start, end = date_range
            changes = self.analyzer.analyze_changes(start, end)
        else:
            changes = self.analyzer.analyze_changes()
        
        return changes
    
    def _query_revenue(
        self,
        date: str = None,
        date_range: Tuple[str, str] = None,
        material_prices: Dict = None,
    ) -> Dict:
        """
        Query revenue estimates.
        
        Args:
            date: Single date
            date_range: (start_date, end_date) tuple
            material_prices: Override material prices
        
        Returns:
            Revenue estimates
        """
        if date:
            return self.estimator.estimate_revenue(date, material_prices)
        elif date_range:
            start, end = date_range
            return self.estimator.estimate_revenue_range(start, end, material_prices)
        else:
            return {'error': 'Must provide date or date_range'}
    
    def _query_timeline(
        self,
        date_range: Tuple[str, str] = None,
        metric: str = 'volume',
    ) -> Dict:
        """
        Query timeline data.
        
        Args:
            date_range: (start_date, end_date) tuple
            metric: Metric to return ('volume', 'area', 'tonnage', 'containers')
        
        Returns:
            Timeline data
        """
        if date_range:
            start, end = date_range
            timeline = self.analyzer.get_inventory_timeline(start, end)
        else:
            timeline = self.analyzer.get_inventory_timeline()
        
        # Extract metric values
        values = []
        dates = []
        for snapshot in timeline:
            dates.append(snapshot['date'])
            if metric == 'volume':
                values.append(snapshot.get('volume_m3', 0))
            elif metric == 'area':
                values.append(snapshot.get('area_m2', 0))
            elif metric == 'tonnage':
                values.append(snapshot.get('tonnage', 0))
            elif metric == 'containers':
                values.append(snapshot.get('containers', 0))
            else:
                values.append(snapshot.get('volume_m3', 0))
        
        return {
            'metric': metric,
            'dates': dates,
            'values': values,
            'timeline': timeline,
        }


if __name__ == "__main__":
    # Test query engine
    engine = QueryEngine()
    
    print("[*] Testing query engine...")
    
    # Test inventory query
    print("\n[+] Inventory query:")
    inv = engine.query('inventory', date='2021-12-01')
    print(f"    Date: {inv.get('date')}")
    print(f"    Volume: {inv.get('volume_m3', 0):.1f} m³")
    print(f"    Tonnage: {inv.get('tonnage', 0):.1f} tons")
    
    # Test containers query
    print("\n[+] Containers query:")
    containers = engine.query('containers', date_range=('2021-01-01', '2022-12-31'))
    print(f"    Total: {containers.get('total_containers', 0)}")
    print(f"    Net change: {containers.get('net_change', 0)}")
    
    # Test changes query
    print("\n[+] Changes query:")
    changes = engine.query('changes', date_range=('2021-01-01', '2022-12-31'))
    print(f"    Change periods: {changes['summary']['total_changes']}")
    print(f"    Arrivals: {changes['summary']['arrivals']}")
    print(f"    Departures: {changes['summary']['departures']}")
    
    # Test revenue query
    print("\n[+] Revenue query:")
    revenue = engine.query('revenue', date='2021-12-01')
    print(f"    Estimated value: ${revenue.get('estimated_value', 0):,.0f}")
    print(f"    Confidence: {revenue.get('confidence', 'unknown')}")
    
    # Test timeline query
    print("\n[+] Timeline query:")
    timeline = engine.query('timeline', metric='volume')
    print(f"    Data points: {len(timeline['dates'])}")
    print(f"    Values: {timeline['values'][:3]}...")
