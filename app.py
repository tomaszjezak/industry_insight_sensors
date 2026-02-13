"""
EDCO Insights Dashboard - AI-Enhanced Visual Monitoring
Inspired by industrial AI monitoring systems with overlays and alerts.
"""

import streamlit as st
import numpy as np
from pathlib import Path
import cv2
from datetime import datetime, date
import plotly.graph_objects as go
import json

# Page config
st.set_page_config(
    page_title="EDCO Insights",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Import components
from src.query_engine import QueryEngine
from src.timelapse import find_timelapse_images, get_timelapse_summary
from src.timeseries_db import TimeseriesDB
from src.pipeline import DebrisAnalysisPipeline


# Clean Professional CSS - White Theme
st.markdown("""
<style>
    .stApp {
        background: #ffffff !important;
        font-family: 'Segoe UI', -apple-system, sans-serif;
    }
    
    .header {
        background: linear-gradient(135deg, #228B22 0%, #32CD32 100%);
        color: white !important;
        padding: 20px 30px;
        margin: -1rem -1rem 2rem -1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .header h1 {
        color: white !important;
    }
    
    /* Ensure all text in main content is dark */
    .main .block-container {
        color: #333333 !important;
    }
    
    /* Streamlit default text should be dark */
    p, div, span, label {
        color: #333333 !important;
    }
    
    .metric-card {
        background: white;
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
    }
    
    .metric-label {
        font-size: 0.75em;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 5px;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def numpy_to_streamlit(img: np.ndarray) -> np.ndarray:
    """Convert BGR to RGB, preserving quality."""
    if img is None:
        return None
    # Ensure we're working with uint8 for best quality
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def draw_ai_overlay(image: np.ndarray, analysis_result: dict = None) -> np.ndarray:
    """
    Draw AI monitoring overlays on image - segmentation outlines for individual piles, bounding boxes, metrics.
    Supports both AI model outputs (with bounding boxes) and traditional segmentation (with masks).
    """
    overlay = image.copy()
    h, w = overlay.shape[:2]
    
    # Check if we have AI-detected objects with bounding boxes
    if analysis_result and 'ai_objects' in analysis_result:
        # Draw AI-detected objects with bounding boxes
        ai_objects = analysis_result['ai_objects']
        for idx, obj in enumerate(ai_objects):
            bbox = obj.get('bbox', [])
            if len(bbox) == 4:
                x, y, width, height = bbox
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + width), int(y + height)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    # Draw green bounding box
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 4)
                    
                    # Add label with AI estimates
                    volume = obj.get('volume_m3', 0)
                    tonnage = obj.get('tonnage_tons', 0)
                    label_text = f"Pile {idx+1}: {tonnage:.1f}t, {volume:.1f}m³"
                    
                    # Position label above box
                    label_y = max(y1 - 10, 20)
                    (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    
                    # White background
                    padding = 5
                    cv2.rectangle(overlay,
                                 (x1, label_y - text_height - baseline - padding),
                                 (x1 + text_width + padding, label_y + baseline + padding),
                                 (255, 255, 255), -1)
                    # Dark text
                    cv2.putText(overlay, label_text, (x1 + padding//2, label_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Also draw segmentation outlines if available (for fallback or combined view)
    elif analysis_result and 'segmentation' in analysis_result:
        seg_result = analysis_result['segmentation']
        if 'mask' in seg_result:
            mask = seg_result['mask']
            if isinstance(mask, np.ndarray) and mask.size > 0:
                # Ensure mask is binary (0 or 255)
                if mask.dtype != np.uint8:
                    mask = (mask > 0).astype(np.uint8) * 255
                elif mask.max() <= 1:
                    mask = (mask * 255).astype(np.uint8)
                
                # Find individual contours (each pile is a separate contour)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Filter out very small contours (noise) and very large ones (buildings)
                # Debris piles are typically 2-30% of image area
                min_area = h * w * 0.02  # At least 2% of image
                max_area = h * w * 0.30  # Max 30% (buildings are larger)
                
                num_piles = 0
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if min_area < area < max_area:
                        num_piles += 1
                        # Draw green outline for each individual pile (thicker for visibility)
                        cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 4)
                        
                        # Add label for each pile
                        M = cv2.moments(contour)
                        if M['m00'] > 0:
                            cx = int(M['m10'] / M['m00'])
                            cy = int(M['m01'] / M['m00'])
                            # Label with area - white background with dark text for visibility
                            area_m2 = area * 0.01  # Rough estimate (would need proper scaling)
                            label_text = f"Pile {num_piles}: {area_m2:.0f}m²"
                            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                            # White background box
                            padding = 5
                            cv2.rectangle(overlay, 
                                         (cx - text_width//2 - padding, cy - text_height - baseline - padding),
                                         (cx + text_width//2 + padding, cy + baseline + padding),
                                         (255, 255, 255), -1)
                            # Dark text (black or dark green)
                            cv2.putText(overlay, label_text, (cx - text_width//2, cy),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Black text on white
    
    # Also check _visualizations for segmentation mask
    elif analysis_result and '_visualizations' in analysis_result:
        seg_vis = analysis_result['_visualizations'].get('segmentation')
        if seg_vis is not None:
            # Extract mask from visualization (green channel shows segmentation)
            seg_gray = cv2.cvtColor(seg_vis, cv2.COLOR_BGR2GRAY)
            # Threshold to get binary mask
            _, mask = cv2.threshold(seg_gray, 50, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            min_area = h * w * 0.02
            max_area = h * w * 0.30
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area < area < max_area:
                    cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 3)
    
    # Draw containers with bounding boxes (blue)
    if analysis_result and 'containers' in analysis_result:
        for container in analysis_result.get('containers', []):
            if 'bbox' in container:
                bbox = np.array(container['bbox'], dtype=np.int32)
                # Draw blue bounding box
                cv2.drawContours(overlay, [bbox], -1, (255, 165, 0), 3)
                
                # Add label with dark background for visibility
                cx, cy = container.get('centroid', (0, 0))
                label = f"Container: {container.get('type', 'unknown')}"
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                # White background for text
                cv2.rectangle(overlay, (cx - 90, cy - 30), (cx + text_width - 80, cy + 5), (255, 255, 255), -1)
                cv2.putText(overlay, label, (cx - 80, cy - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)  # Black text
    
    # Draw metrics overlay (green boxes in corner with clear labels)
    if analysis_result:
        metrics = []
        if 'volume' in analysis_result:
            vol = analysis_result['volume'].get('volume_m3', 0)
            metrics.append(("Debris Volume", f"{vol:.0f} m³"))
        if 'volume' in analysis_result:
            tons = analysis_result['volume'].get('tonnage_estimate', 0)
            metrics.append(("Material Tonnage", f"{tons:.1f} tons"))
        
        y_pos = h - 80
        for label, value in metrics:
            text = f"{label}: {value}"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            # Green background box with dark text for visibility
            cv2.rectangle(overlay, (w - text_width - 30, y_pos - 25),
                         (w - 10, y_pos + 5), (200, 255, 200), -1)  # Light green background
            cv2.rectangle(overlay, (w - text_width - 30, y_pos - 25),
                         (w - 10, y_pos + 5), (34, 139, 34), 2)  # Green border
            cv2.putText(overlay, text, (w - text_width - 20, y_pos),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 0), 2)  # Dark green text
            y_pos -= 35
    
    return overlay


def calculate_purity_score(materials: dict) -> float:
    """Calculate material purity score."""
    if not materials:
        return 0
    mixed_pct = materials.get('mixed', 0)
    dominant_pct = max(materials.values()) if materials else 0
    return max(dominant_pct, 100 - mixed_pct)


def get_image_stats(img_path: Path, img_date: datetime, db_images: list, db: TimeseriesDB) -> dict:
    """Get analysis stats for a specific image."""
    img_date_str = img_date.date().isoformat()
    db_img = None
    for db_img_row in db_images:
        if db_img_row['date'].startswith(img_date_str):
            db_img = db_img_row
            break
    
    stats = {
        'container_count': 0,
        'tonnage': 0.0,
        'volume': 0.0,
        'purity': 0.0,
        'materials': {},
        'has_data': False
    }
    
    if db_img:
        stats['has_data'] = True
        try:
            containers = db.get_containers_by_date_range(img_date_str, img_date_str)
            container_list = [c for c in containers if c['image_id'] == db_img['id']] if containers else []
            stats['container_count'] = int(len(container_list)) if container_list else 0
        except:
            stats['container_count'] = 0
        
        tonnage_raw = db_img.get('tonnage_estimate', 0) or 0
        try:
            stats['tonnage'] = float(tonnage_raw) if tonnage_raw is not None else 0.0
        except (ValueError, TypeError):
            stats['tonnage'] = 0.0
        
        volume_raw = db_img.get('volume_m3', 0) or 0
        try:
            stats['volume'] = float(volume_raw) if volume_raw is not None else 0.0
        except (ValueError, TypeError):
            stats['volume'] = 0.0
        
        if db_img.get('materials_json'):
            try:
                stats['materials'] = json.loads(db_img['materials_json'])
                purity_raw = calculate_purity_score(stats['materials'])
                stats['purity'] = float(purity_raw) if purity_raw is not None else 0.0
            except:
                stats['purity'] = 0.0
                stats['materials'] = {}
    
    return stats


def render_image_grid(images_in_range: list, db_images: list, db: TimeseriesDB):
    """Render image grid with inline details - all images in one row."""
    st.markdown("### 📸 Facility Images")
    
    # All images in one row - 7 columns
    num_images = len(images_in_range)
    cols = st.columns(num_images)
    
    for idx, (img_path, img_date) in enumerate(images_in_range):
        with cols[idx]:
            # Load and resize thumbnail - smaller to fit 7 in one row
            thumb = cv2.imread(str(img_path))
            if thumb is not None:
                # Smaller thumbnails - max 200px to fit 7 in one row
                h, w = thumb.shape[:2]
                max_dim = 200
                if w > h:
                    new_w = max_dim
                    new_h = int(h * (max_dim / w))
                else:
                    new_h = max_dim
                    new_w = int(w * (max_dim / h))
                
                thumb_resized = cv2.resize(thumb, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                date_str = img_date.strftime('%b %Y')  # Shorter format: "Feb 2023"
                
                # Display image with date as caption
                st.image(numpy_to_streamlit(thumb_resized), 
                        caption=f"**{date_str}**", 
                        use_container_width=True)
    
    st.markdown("---")
    
    # Show details for all images inline - no modal needed
    st.markdown("### 📊 Image Details")
    
    # Display each image with its stats in expandable sections
    for idx, (img_path, img_date) in enumerate(images_in_range):
        date_str = img_date.strftime('%B %Y')
        
        with st.expander(f"🔍 {date_str} - View Details", expanded=(idx == len(images_in_range) - 1)):
            # Get stats
            stats = get_image_stats(img_path, img_date, db_images, db)
            
            # Load FULL HD image directly from file (not thumbnail)
            image = cv2.imread(str(img_path))
            if image is not None:
                # Get analysis result for segmentation
                img_date_str = img_date.date().isoformat()
                db_img = None
                for db_img_row in db_images:
                    if db_img_row['date'].startswith(img_date_str):
                        db_img = db_img_row
                        break
                
                analysis_result = None
                container_list = []
                if db_img:
                    try:
                        containers = db.get_containers_by_date_range(img_date_str, img_date_str)
                        container_list = [c for c in containers if c['image_id'] == db_img['id']] if containers else []
                    except:
                        container_list = []
                    
                    analysis_result = {
                        'volume': {
                            'volume_m3': db_img.get('volume_m3', 0) or 0,
                            'tonnage_estimate': db_img.get('tonnage_estimate', 0) or 0,
                        },
                        'containers': container_list,
                        'segmentation': {'mask': None}
                    }
                
                # Use AI multimodal model for analysis (if API key configured)
                use_ai = st.session_state.get('use_ai_analysis', False)
                ai_provider = st.session_state.get('ai_provider', 'openai')
                ai_api_key = st.session_state.get('ai_api_key', None)
                
                if use_ai and ai_api_key:
                    try:
                        from src.ai_analyzer import AIRecyclingAnalyzer
                        ai_analyzer = AIRecyclingAnalyzer(
                            provider=ai_provider,
                            api_key=ai_api_key
                        )
                        ai_result = ai_analyzer.analyze(image, camera_height_m=4.0)
                        
                        # Use AI results
                        if ai_result and 'segmentation' in ai_result:
                            if analysis_result is None:
                                analysis_result = {}
                            analysis_result['segmentation'] = ai_result['segmentation']
                            analysis_result['volume'] = ai_result.get('volume', {})
                            analysis_result['materials'] = ai_result.get('materials', {})
                            analysis_result['ai_objects'] = ai_result.get('objects', [])
                    except Exception as e:
                        st.warning(f"AI analysis failed: {e}. Falling back to OpenCV segmentation.")
                        use_ai = False
                
                if not use_ai or not ai_api_key:
                    # Fallback to fast OpenCV segmentation
                    from src.segmentation import PileSegmenter
                    segmenter = PileSegmenter(use_sam=False)
                    seg_result = segmenter.segment(image)
                    if seg_result and 'mask' in seg_result and seg_result['mask'] is not None:
                        if analysis_result is None:
                            analysis_result = {'segmentation': {}}
                        analysis_result['segmentation']['mask'] = seg_result['mask']
                
                # Draw overlays on FULL HD image (image is already full resolution from cv2.imread)
                overlay_image = draw_ai_overlay(image, analysis_result)
                
                # Display FULL HD image with overlays - use original file path for best quality
                # This ensures full resolution when user clicks fullscreen
                st.image(numpy_to_streamlit(overlay_image), 
                        caption=f"AI-Enhanced View - {date_str} (Full HD)",
                        use_container_width=True)
                
                # Also provide original file for maximum quality fullscreen viewing
                with st.expander("🔍 View Original Full Resolution (Click to Fullscreen)"):
                    st.image(str(img_path),
                            caption=f"Original - {date_str}",
                            use_container_width=False)
                
                # Stats in columns
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Material Tonnage", f"{stats['tonnage']:.1f} tons")
                with col2:
                    st.metric("Debris Volume", f"{stats['volume']:.0f} m³")
                with col3:
                    st.metric("Containers", stats['container_count'])
                with col4:
                    st.metric("Purity", f"{stats['purity']:.0f}%")
                
                # Material breakdown
                if stats['materials']:
                    st.markdown("**Material Breakdown:**")
                    for material, pct in stats['materials'].items():
                        st.progress(pct / 100, text=f"{material.capitalize()}: {pct:.1f}%")


def main():
    # Header
    st.markdown("""
    <div class="header">
        <h1 style="margin:0; font-size: 2em; display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 1.5em;">♻️</span> EDCO Insights - Site 38782
            <span style="margin-left: auto; font-size: 0.6em; opacity: 0.9;">AI-Powered Monitoring</span>
        </h1>
    </div>
    """, unsafe_allow_html=True)
    
    # AI Analysis Configuration (Sidebar) - MUST be after header for Streamlit to show it
    with st.sidebar:
        st.markdown("### 🤖 AI Analysis")
        st.markdown("**Configure AI model for recycling detection**")
        use_ai = st.checkbox("Use AI Model (GPT-4V/Claude/Gemini)", value=False, key="use_ai_analysis")
        if use_ai:
            ai_provider = st.selectbox("Provider", ["google", "openai", "anthropic"], key="ai_provider", 
                                      help="Google Gemini is FREE! Get API key: https://aistudio.google.com/apikey")
            ai_api_key = st.text_input("API Key", type="password", key="ai_api_key", 
                                      help="Google Gemini: FREE at https://aistudio.google.com/apikey | OpenAI: https://platform.openai.com/api-keys | Anthropic: https://console.anthropic.com/")
            if ai_api_key:
                st.session_state['ai_api_key'] = ai_api_key
                st.session_state['ai_provider'] = ai_provider
                st.success("✓ API key configured")
            else:
                st.warning("⚠️ Enter API key to use AI analysis")
        else:
            st.info("💡 Enable AI model to get accurate pile detection, volume estimates, and material classification")
    
    # Initialize
    timelapse_summary = get_timelapse_summary()
    engine = QueryEngine()
    
    if timelapse_summary['total_images'] == 0:
        st.warning("No timelapse data. Run: python -m src.batch_processor")
        return
    
    # Date Range
    if timelapse_summary['date_range']:
        start_date = datetime.fromisoformat(timelapse_summary['date_range']['start']).date()
        end_date = datetime.fromisoformat(timelapse_summary['date_range']['end']).date()
    else:
        start_date = date(2015, 1, 1)
        end_date = date.today()
    
    col_d1, col_d2 = st.columns([1, 4])
    with col_d1:
        st.markdown("### 📅 Date Range")
    with col_d2:
        col_da, col_db = st.columns(2)
        with col_da:
            selected_start = st.date_input("From", value=start_date, min_value=start_date, max_value=end_date, key="date_start", label_visibility="collapsed")
        with col_db:
            selected_end = st.date_input("To", value=end_date, min_value=selected_start if isinstance(selected_start, date) else start_date, max_value=end_date, key="date_end", label_visibility="collapsed")
    
    if isinstance(selected_start, tuple):
        selected_start = selected_start[0]
    if isinstance(selected_end, tuple):
        selected_end = selected_end[0]
    
    # Query data
    inventory = engine.query('inventory', date_range=(selected_start.isoformat(), selected_end.isoformat()))
    changes = engine.query('changes', date_range=(selected_start.isoformat(), selected_end.isoformat()))
    timeline = engine.query('timeline', date_range=(selected_start.isoformat(), selected_end.isoformat()))
    
    snapshots = inventory.get('snapshots', [])
    timeline_data = timeline.get('timeline', [])
    
    if not snapshots:
        st.info("No data for selected date range")
        return
    
    latest = snapshots[-1]
    
    # Get analysis result for latest image
    db = TimeseriesDB()
    db_images = db.get_all_images()  # Renamed to avoid conflict
    latest_img = None
    for db_img in db_images:
        if db_img['date'].startswith(selected_end.isoformat()[:10]):
            latest_img = db_img
            break
    
    # Get images in date range
    timelapse_images = find_timelapse_images()
    images_in_range = [
        (img_path, dt) for img_path, dt in timelapse_images
        if selected_start <= dt.date() <= selected_end
    ]
    
    if images_in_range:
        # Render image grid with inline details (no modal)
        render_image_grid(images_in_range, db_images, db)
    else:
        st.info("No images found in the selected date range.")


if __name__ == "__main__":
    main()
