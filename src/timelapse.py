"""
Time-lapse analysis module for processing dated Street View images.
Extracts dates from filenames and batch processes images through the pipeline.
"""

import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import cv2


# Month name to number mapping
MONTH_MAP = {
    'jan': 1, 'january': 1,
    'feb': 2, 'february': 2,
    'mar': 3, 'march': 3,
    'apr': 4, 'april': 4,
    'may': 5,
    'jun': 6, 'june': 6,
    'jul': 7, 'july': 7,
    'aug': 8, 'august': 8,
    'sep': 9, 'september': 9,
    'oct': 10, 'october': 10,
    'nov': 11, 'november': 11,
    'dec': 12, 'december': 12,
}


def extract_date_from_filename(filename: str) -> Optional[datetime]:
    """
    Extract date from EDCO filename format.
    
    Examples:
        edco_feb2015_federal.png -> 2015-02-01
        edco_dec2021_federal.png -> 2021-12-01
        edo_feb2021_federal.png -> 2021-02-01
    
    Args:
        filename: Filename like "edco_feb2015_federal.png"
    
    Returns:
        datetime object or None if parsing fails
    """
    filename_lower = filename.lower()
    
    # Pattern: month_name + year (e.g., "feb2015", "dec2021")
    # Match month name (3+ letters) followed by 4-digit year
    pattern = r'([a-z]{3,})\s*(\d{4})'
    match = re.search(pattern, filename_lower)
    
    if not match:
        return None
    
    month_str, year_str = match.groups()
    year = int(year_str)
    
    # Get month number
    month = MONTH_MAP.get(month_str)
    if not month:
        return None
    
    # Return first day of month (we don't have exact day from filename)
    return datetime(year, month, 1)


def find_timelapse_images(data_dir: Path = None) -> List[Tuple[Path, datetime]]:
    """
    Find all timelapse images and extract their dates.
    
    Args:
        data_dir: Directory containing timelapse images. Defaults to data/edco_timelapse/
    
    Returns:
        List of (image_path, datetime) tuples, sorted by date
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "edco_timelapse"
    
    if not data_dir.exists():
        return []
    
    images_with_dates = []
    
    for img_path in data_dir.glob("*.png"):
        date = extract_date_from_filename(img_path.name)
        if date:
            images_with_dates.append((img_path, date))
        else:
            print(f"[!] Could not extract date from: {img_path.name}")
    
    # Sort by date
    images_with_dates.sort(key=lambda x: x[1])
    
    return images_with_dates


def validate_image(image_path: Path) -> bool:
    """
    Validate that image can be loaded.
    
    Args:
        image_path: Path to image file
    
    Returns:
        True if image is valid, False otherwise
    """
    try:
        img = cv2.imread(str(image_path))
        return img is not None and img.size > 0
    except Exception as e:
        print(f"[!] Error loading {image_path}: {e}")
        return False


def get_timelapse_summary(data_dir: Path = None) -> Dict:
    """
    Get summary of available timelapse images.
    
    Args:
        data_dir: Directory containing timelapse images
    
    Returns:
        Dict with summary statistics
    """
    images = find_timelapse_images(data_dir)
    
    if not images:
        return {
            'total_images': 0,
            'date_range': None,
            'images': [],
        }
    
    dates = [dt for _, dt in images]
    
    return {
        'total_images': len(images),
        'date_range': {
            'start': dates[0].isoformat(),
            'end': dates[-1].isoformat(),
            'span_days': (dates[-1] - dates[0]).days,
            'span_years': (dates[-1] - dates[0]).days / 365.25,
        },
        'images': [
            {
                'filename': img_path.name,
                'date': dt.isoformat(),
                'path': str(img_path),
            }
            for img_path, dt in images
        ],
    }


if __name__ == "__main__":
    # Test date extraction
    test_files = [
        "edco_feb2015_federal.png",
        "edco_apr2018_federal.png",
        "edco_feb2019_federal.png",
        "edo_feb2021_federal.png",
        "edco_dec2021_federal.png",
        "edco_may2022_federal.png",
        "edco_jan2023_federal.png",
    ]
    
    print("Testing date extraction:")
    for filename in test_files:
        date = extract_date_from_filename(filename)
        if date:
            print(f"  {filename:35} -> {date.strftime('%Y-%m-%d')}")
        else:
            print(f"  {filename:35} -> FAILED")
    
    print("\nFinding timelapse images:")
    images = find_timelapse_images()
    print(f"Found {len(images)} images")
    for img_path, date in images:
        print(f"  {date.strftime('%Y-%m-%d')}: {img_path.name}")
    
    print("\nSummary:")
    summary = get_timelapse_summary()
    print(f"  Total: {summary['total_images']}")
    if summary['date_range']:
        print(f"  Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        print(f"  Span: {summary['date_range']['span_years']:.1f} years")
