"""
SQLite database for storing time-series analysis results.
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import json


class TimeseriesDB:
    """SQLite database for time-series data."""
    
    def __init__(self, db_path: Path = None):
        """
        Initialize database.
        
        Args:
            db_path: Path to SQLite database file. Defaults to data/timeseries/timelapse.db
        """
        if db_path is None:
            db_path = Path(__file__).parent.parent / "data" / "timeseries" / "timelapse.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_schema()
    
    def _init_schema(self):
        """Initialize database schema."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Images table - metadata about each image
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL UNIQUE,
                filepath TEXT NOT NULL,
                date TEXT NOT NULL,
                camera_height_m REAL,
                image_width INTEGER,
                image_height INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Analysis results table - main analysis output
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                volume_m3 REAL,
                pile_area_m2 REAL,
                avg_height_m REAL,
                max_height_m REAL,
                tonnage_estimate REAL,
                materials_json TEXT,
                dominant_material TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (image_id) REFERENCES images(id)
            )
        """)
        
        # Containers table - detected containers
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS containers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                container_type TEXT,
                length_m REAL,
                width_m REAL,
                centroid_x INTEGER,
                centroid_y INTEGER,
                confidence REAL,
                bbox_json TEXT,
                FOREIGN KEY (image_id) REFERENCES images(id)
            )
        """)
        
        # Pallets table - detected pallets/bales
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pallets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                size_m REAL,
                centroid_x INTEGER,
                centroid_y INTEGER,
                confidence REAL,
                bbox_json TEXT,
                FOREIGN KEY (image_id) REFERENCES images(id)
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_date ON images(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_image_id ON analysis_results(image_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_containers_image_id ON containers(image_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pallets_image_id ON pallets(image_id)")
        
        conn.commit()
        conn.close()
    
    def add_image(
        self,
        filename: str,
        filepath: str,
        date: datetime,
        camera_height_m: float = None,
        image_width: int = None,
        image_height: int = None,
    ) -> int:
        """
        Add image metadata.
        
        Returns:
            Image ID
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO images 
            (filename, filepath, date, camera_height_m, image_width, image_height)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            filename,
            str(filepath),
            date.isoformat(),
            camera_height_m,
            image_width,
            image_height,
        ))
        
        image_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return image_id
    
    def add_analysis_result(
        self,
        image_id: int,
        volume_m3: float,
        pile_area_m2: float,
        avg_height_m: float,
        max_height_m: float,
        tonnage_estimate: float,
        materials: Dict,
        dominant_material: str,
    ):
        """Add analysis result."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO analysis_results
            (image_id, volume_m3, pile_area_m2, avg_height_m, max_height_m, 
             tonnage_estimate, materials_json, dominant_material)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            image_id,
            volume_m3,
            pile_area_m2,
            avg_height_m,
            max_height_m,
            tonnage_estimate,
            json.dumps(materials),
            dominant_material,
        ))
        
        conn.commit()
        conn.close()
    
    def add_containers(self, image_id: int, containers: List[Dict]):
        """Add detected containers."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        for container in containers:
            cursor.execute("""
                INSERT INTO containers
                (image_id, container_type, length_m, width_m, centroid_x, centroid_y, 
                 confidence, bbox_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                image_id,
                container.get('type'),
                container.get('length_m'),
                container.get('width_m'),
                container['centroid'][0],
                container['centroid'][1],
                container.get('confidence', 0.5),
                json.dumps(container.get('bbox', [])),
            ))
        
        conn.commit()
        conn.close()
    
    def add_pallets(self, image_id: int, pallets: List[Dict]):
        """Add detected pallets."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        for pallet in pallets:
            cursor.execute("""
                INSERT INTO pallets
                (image_id, size_m, centroid_x, centroid_y, confidence, bbox_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                image_id,
                pallet.get('size_m'),
                pallet['centroid'][0],
                pallet['centroid'][1],
                pallet.get('confidence', 0.5),
                json.dumps(pallet.get('bbox', [])),
            ))
        
        conn.commit()
        conn.close()
    
    def get_all_images(self) -> List[Dict]:
        """Get all images with their analysis results."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT i.*, a.volume_m3, a.pile_area_m2, a.tonnage_estimate, 
                   a.materials_json, a.dominant_material
            FROM images i
            LEFT JOIN analysis_results a ON i.id = a.image_id
            ORDER BY i.date
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_containers_by_date_range(self, start_date: str, end_date: str) -> List[Dict]:
        """Get containers in date range."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT c.*, i.date, i.filename
            FROM containers c
            JOIN images i ON c.image_id = i.id
            WHERE i.date >= ? AND i.date <= ?
            ORDER BY i.date
        """, (start_date, end_date))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM images")
        image_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM containers")
        container_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM pallets")
        pallet_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT MIN(date), MAX(date) FROM images")
        date_range = cursor.fetchone()
        
        conn.close()
        
        return {
            'total_images': image_count,
            'total_containers': container_count,
            'total_pallets': pallet_count,
            'date_range': {
                'start': date_range[0],
                'end': date_range[1],
            } if date_range[0] else None,
        }


if __name__ == "__main__":
    # Test database
    db = TimeseriesDB()
    print(f"Database initialized at: {db.db_path}")
    
    stats = db.get_summary_stats()
    print(f"Current stats: {stats}")
