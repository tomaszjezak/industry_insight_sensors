"""
Image scraper for C&D debris pile images.
Downloads sample images for pipeline testing and development.
"""

import os
from pathlib import Path
import requests
from urllib.parse import urlparse
import time

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "rgb_images"

# Sample URLs for C&D debris images - AERIAL/OVERHEAD views needed!
# Sources: Shutterstock previews, Dreamstime, Alamy previews (for dev/prototyping only)
SAMPLE_IMAGE_URLS = [
    # Aerial views of recycling yards and debris piles
    "https://thumbs.dreamstime.com/z/waste-disposal-pile-plastic-garbage-various-trash-aerial-view-drone-recycling-junk-construction-pollution-concept-179395580.jpg",
    "https://thumbs.dreamstime.com/z/landfill-construction-waste-cdw-trash-disposal-facility-aerial-drone-view-recycling-junk-pollution-concept-179395469.jpg",
    "https://thumbs.dreamstime.com/z/aerial-view-garbage-dump-landfill-trash-disposal-site-environmental-pollution-ecology-concept-122527539.jpg",
    "https://thumbs.dreamstime.com/z/aerial-view-drone-scrap-yard-car-junkyard-old-cars-being-dismantled-parts-metal-recycling-rusty-wrecked-automobiles-146908665.jpg",
    "https://thumbs.dreamstime.com/z/aerial-view-industrial-wasteland-dumping-site-heap-waste-construction-garbage-top-down-drone-shot-171562877.jpg",
    "https://thumbs.dreamstime.com/z/overhead-view-construction-site-aerial-photo-demolition-work-concrete-debris-pile-rubble-excavator-200431877.jpg",
    "https://thumbs.dreamstime.com/z/aerial-top-view-scrap-metal-junkyard-waste-recycling-facility-pile-rusty-old-cars-auto-parts-190890158.jpg",
    "https://thumbs.dreamstime.com/z/landfill-recycling-construction-waste-dispose-debris-industrial-rubbish-treatment-processing-factory-garbage-dump-138741395.jpg",
    "https://thumbs.dreamstime.com/z/aerial-view-city-dump-large-garbage-pile-waste-sorting-plant-environmental-pollution-155241789.jpg",
    "https://thumbs.dreamstime.com/z/aerial-view-recycling-plant-large-pile-waste-garbage-dump-environmental-pollution-city-industrial-site-155241858.jpg",
    # More overhead debris pile views
    "https://thumbs.dreamstime.com/z/construction-site-demolition-aerial-view-building-debris-pile-concrete-rubble-excavation-work-top-down-200431756.jpg",
    "https://thumbs.dreamstime.com/z/aerial-top-down-view-construction-demolition-site-debris-pile-concrete-rubble-bricks-195821545.jpg",
]


def download_images(
    urls: list[str] = None,
    limit: int = 8,
    output_dir: Path = None,
) -> list[Path]:
    """
    Download images from URLs.
    
    Args:
        urls: List of image URLs. Defaults to SAMPLE_IMAGE_URLS.
        limit: Maximum number of images to download.
        output_dir: Directory to save images. Defaults to DATA_DIR.
    
    Returns:
        List of paths to downloaded images.
    """
    urls = (urls or SAMPLE_IMAGE_URLS)[:limit]
    output_dir = output_dir or DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded = []
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
    }
    
    for i, url in enumerate(urls):
        print(f"[{i+1}/{len(urls)}] Downloading: {url[:60]}...")
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Determine filename
            content_type = response.headers.get('content-type', '')
            if 'jpeg' in content_type or 'jpg' in content_type:
                ext = '.jpg'
            elif 'png' in content_type:
                ext = '.png'
            elif 'webp' in content_type:
                ext = '.webp'
            else:
                ext = '.jpg'  # Default
            
            filename = f"debris_{i+1:03d}{ext}"
            filepath = output_dir / filename
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            downloaded.append(filepath)
            print(f"    ✓ Saved: {filename}")
            
            # Small delay to be nice to servers
            time.sleep(0.3)
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    print(f"\n[+] Downloaded {len(downloaded)} images to {output_dir}")
    return downloaded


def flatten_images(output_dir: Path = None) -> list[Path]:
    """
    Flatten nested query folders into a single directory with renamed files.
    
    Args:
        output_dir: Directory containing downloaded images.
    
    Returns:
        List of paths to flattened images.
    """
    output_dir = output_dir or DATA_DIR
    flattened = []
    counter = 1
    
    for query_dir in output_dir.iterdir():
        if query_dir.is_dir():
            for img_path in query_dir.glob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                    # Create new filename
                    new_name = f"debris_{counter:03d}{img_path.suffix.lower()}"
                    new_path = output_dir / new_name
                    
                    # Move file
                    img_path.rename(new_path)
                    flattened.append(new_path)
                    counter += 1
            
            # Remove empty query directory
            try:
                query_dir.rmdir()
            except OSError:
                pass  # Directory not empty
    
    print(f"[+] Flattened {len(flattened)} images to {output_dir}")
    return flattened


def list_available_images(data_dir: Path = None) -> list[Path]:
    """List all available images in the data directory."""
    data_dir = data_dir or DATA_DIR
    
    if not data_dir.exists():
        return []
    
    images = []
    for pattern in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        images.extend(data_dir.glob(pattern))
        # Also check subdirectories
        images.extend(data_dir.glob(f"**/{pattern}"))
    
    return sorted(images)


if __name__ == "__main__":
    print("=" * 60)
    print("C&D Debris Image Scraper")
    print("=" * 60)
    
    # Download images
    downloaded = download_images(limit=5)
    
    if downloaded:
        # Flatten into single directory
        flatten_images()
    
    # List what we have
    available = list_available_images()
    print(f"\n[*] Available images: {len(available)}")
    for img in available[:10]:
        print(f"    - {img.name}")
    if len(available) > 10:
        print(f"    ... and {len(available) - 10} more")
