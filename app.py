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
    """
    overlay = image.copy()
    h, w = overlay.shape[:2]
    
    # Draw segmentation outlines for individual debris piles (green contours)
    if analysis_result and 'segmentation' in analysis_result:
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
    
    # Main Image with AI Overlays
    st.markdown("### 🎯 Live Site Monitoring")
    
    timelapse_images = find_timelapse_images()  # Renamed to avoid conflict
    images_in_range = [
        (img_path, dt) for img_path, dt in timelapse_images
        if selected_start <= dt.date() <= selected_end
    ]
    
    if images_in_range:
        # Image Gallery - Show all thumbnails
        st.markdown("#### 📸 Image Gallery")
        num_cols = 4
        cols = st.columns(num_cols)
        
        for idx, (img_path, img_date) in enumerate(images_in_range):
            col_idx = idx % num_cols
            with cols[col_idx]:
                # Load and resize thumbnail with high quality
                thumb = cv2.imread(str(img_path))
                if thumb is not None:
                    # Preserve aspect ratio, make larger thumbnails
                    h, w = thumb.shape[:2]
                    max_dim = 400  # Larger thumbnails for better quality
                    if w > h:
                        new_w = max_dim
                        new_h = int(h * (max_dim / w))
                    else:
                        new_h = max_dim
                        new_w = int(w * (max_dim / h))
                    
                    # Use high-quality interpolation
                    thumb_resized = cv2.resize(thumb, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                    st.image(numpy_to_streamlit(thumb_resized), 
                            caption=img_date.strftime('%B %Y'), 
                            use_container_width=True)
        
        st.markdown("---")
        
        # Image selector - scroll through images
        st.markdown("#### 🔍 Detailed View")
        if len(images_in_range) > 1:
            selected_idx = st.slider(
                "Select Image",
                0, 
                len(images_in_range) - 1,
                len(images_in_range) - 1,  # Default to latest
                format="Image %d of %d"
            )
        else:
            selected_idx = 0
        
        img_path, img_date = images_in_range[selected_idx]
        
        # Load original full-resolution image directly from file
        image = cv2.imread(str(img_path))
        
        if image is not None:
            # Get analysis for this specific image
            img_date_str = img_date.date().isoformat()
            db_img = None
            for db_img_row in db_images:  # Use renamed variable
                if db_img_row['date'].startswith(img_date_str):
                    db_img = db_img_row
                    break
            
            # Use cached results from database - NO SLOW MODEL LOADING!
            analysis_result = None
            container_list = []  # Initialize to avoid undefined variable
            container_count = 0  # Initialize
            tonnage = 0.0  # Initialize
            volume = 0.0  # Initialize
            purity = 0.0  # Initialize
            
            if db_img:
                # Get containers from database
                try:
                    containers = db.get_containers_by_date_range(img_date_str, img_date_str)
                    container_list = [c for c in containers if c['image_id'] == db_img['id']] if containers else []
                except:
                    container_list = []
                
                # Create result from database (already processed)
                analysis_result = {
                    'volume': {
                        'volume_m3': db_img.get('volume_m3', 0) or 0,
                        'tonnage_estimate': db_img.get('tonnage_estimate', 0) or 0,
                    },
                    'containers': container_list,
                    'segmentation': {
                        'mask': None,  # We'll get this from a simple segmentation if needed
                    }
                }
                
                # Pre-calculate all metric values to ensure they're numeric
                container_count = int(len(container_list)) if container_list else 0
                
                tonnage_raw = db_img.get('tonnage_estimate', 0) or 0
                try:
                    tonnage = float(tonnage_raw) if tonnage_raw is not None else 0.0
                except (ValueError, TypeError):
                    tonnage = 0.0
                
                volume_raw = db_img.get('volume_m3', 0) or 0
                try:
                    volume = float(volume_raw) if volume_raw is not None else 0.0
                except (ValueError, TypeError):
                    volume = 0.0
                
                if db_img.get('materials_json'):
                    try:
                        materials = json.loads(db_img['materials_json'])
                        purity_raw = calculate_purity_score(materials)
                        purity = float(purity_raw) if purity_raw is not None else 0.0
                    except:
                        purity = 0.0
            
            # Always run segmentation for visualization (fast OpenCV method)
            # This ensures we always have segmentation outlines
            from src.segmentation import PileSegmenter
            segmenter = PileSegmenter(use_sam=False)  # Fast OpenCV method
            seg_result = segmenter.segment(image)
            if seg_result and 'mask' in seg_result and seg_result['mask'] is not None:
                if analysis_result is None:
                    analysis_result = {'segmentation': {}}
                analysis_result['segmentation']['mask'] = seg_result['mask']
            
            # Draw AI overlays (segmentation outlines, containers, metrics)
            overlay_image = draw_ai_overlay(image, analysis_result)
            
            col_img1, col_img2 = st.columns([2, 1])
            
            with col_img1:
                date_str = img_date.strftime('%B %Y')
                # Display AI-enhanced image
                st.image(numpy_to_streamlit(overlay_image), 
                        caption=f"AI-Enhanced View - {date_str}", 
                        use_container_width=True)
                
                # Also show original full-resolution image in expander
                # This uses the file path directly, so fullscreen will be full resolution
                with st.expander("🔍 View Original Full Resolution (Click to Fullscreen)"):
                    # Use file path directly - Streamlit will load original for fullscreen
                    st.image(str(img_path),
                            caption=f"Original - {date_str}",
                            use_container_width=False)  # Don't constrain for full res
            
            with col_img2:
                st.markdown("#### 📊 Real-Time Metrics")
                
                if db_img:
                    # All values are pre-calculated and guaranteed to be numeric
                    st.metric("Containers Detected", container_count)
                    st.metric("Material Tonnage", f"{tonnage:.1f} tons")
                    st.metric("Debris Volume", f"{volume:.0f} m³")
                    if purity > 0:
                        st.metric("Material Purity", f"{purity:.0f}%")
                else:
                    st.info("No analysis data for this image")
    
    # Material Quality Section
    st.markdown("---")
    st.markdown("### 🧱 Material Analysis")
    
    if latest_img and latest_img.get('materials_json'):
        materials = json.loads(latest_img['materials_json'])
        
        col_m1, col_m2 = st.columns([2, 1])
        
        with col_m1:
            st.markdown("**Composition Breakdown:**")
            for material, pct in materials.items():
                st.progress(pct / 100, text=f"{material.capitalize()}: {pct:.1f}%")
        
        with col_m2:
            purity = calculate_purity_score(materials)
            
            st.metric("Material Purity", f"{purity:.0f}%")
            if purity >= 85:
                st.success("✓ High purity - well sorted")
            elif purity >= 70:
                st.info("Moderate purity")
            else:
                st.warning("⚠️ Low purity - needs sorting")
    
    # Operational Metrics
    st.markdown("---")
    st.markdown("### ⚙️ Operational Efficiency")
    
    if timeline_data and len(timeline_data) >= 2:
        first = timeline_data[0]
        last = timeline_data[-1]
        days = (datetime.fromisoformat(last['date']) - datetime.fromisoformat(first['date'])).days
        
        if days > 0:
            throughput = (last.get('tonnage', 0) - first.get('tonnage', 0)) / days
        else:
            throughput = 0
        
        col_e1, col_e2, col_e3, col_e4 = st.columns(4)
        
        with col_e1:
            # Ensure throughput is numeric
            throughput_val = float(throughput) if throughput is not None else 0.0
            st.metric("Throughput", f"{throughput_val:.1f} tons/day")
        
        with col_e2:
            arrivals = changes['summary'].get('arrivals', 0) if changes.get('summary') else 0
            arrivals_val = int(arrivals) if arrivals is not None else 0
            st.metric("Arrivals", arrivals_val)
        
        with col_e3:
            departures = changes['summary'].get('departures', 0) if changes.get('summary') else 0
            departures_val = int(departures) if departures is not None else 0
            st.metric("Departures", departures_val)
        
        with col_e4:
            net = changes['summary'].get('net_change', 0) if changes.get('summary') else 0
            # Ensure delta is numeric, not string or None
            net_val = int(net) if net is not None else 0
            # Only pass delta if it's a valid number
            if net is not None and net != 0:
                net_delta = float(net)
                st.metric("Net Change", net_val, delta=net_delta)
            else:
                st.metric("Net Change", net_val)
    
    # Trend Chart
    st.markdown("---")
    st.markdown("### 📈 Volume Trend Analysis")
    
    if timeline.get('values') and len(timeline['values']) >= 2:
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            current = timeline['values'][-1]
            current_val = float(current) if current is not None else 0.0
            st.metric("Current Debris Volume", f"{current_val:.0f} m³")
        with col_t2:
            recent_trend = timeline['values'][-1] - timeline['values'][-2]
            recent_trend_val = float(recent_trend) if recent_trend is not None else 0.0
            # Only pass delta if it's a valid number
            if recent_trend is not None and abs(recent_trend) > 100:
                volume_delta = float(recent_trend)
                st.metric("Volume Change", f"{recent_trend_val:.0f} m³", delta=volume_delta)
            else:
                st.metric("Volume Change", f"{recent_trend_val:.0f} m³")
        with col_t3:
            if len(timeline['values']) >= 2:
                first_vol = timeline['values'][0]
                last_vol = timeline['values'][-1]
                total_change = ((last_vol - first_vol) / first_vol * 100) if first_vol > 0 else 0
                # Ensure all values are numeric
                total_change_val = float(total_change) if total_change is not None else 0.0
                # Only pass delta if it's a valid number
                if total_change is not None and abs(total_change) > 0.1:
                    change_delta = float(total_change)
                    st.metric("Total Change", f"{total_change_val:+.1f}%", delta=change_delta)
                else:
                    st.metric("Total Change", f"{total_change_val:+.1f}%")
        
        # Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timeline['dates'],
            y=timeline['values'],
            mode='lines+markers',
            name='Volume',
            line=dict(color='#228B22', width=3),
            marker=dict(size=8, color='#228B22'),
            fill='tonexty',
            fillcolor='rgba(34, 139, 34, 0.1)',
        ))
        
        fig.update_layout(
            height=400,
            paper_bgcolor='white',
            plot_bgcolor='#f8f9fa',
            font=dict(color='#333'),
            showlegend=False,
            xaxis=dict(gridcolor='#e0e0e0', title='Date'),
            yaxis=dict(gridcolor='#e0e0e0', title='Debris Volume (m³)'),
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


if __name__ == "__main__":
    main()
