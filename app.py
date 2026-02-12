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


# AI Monitoring Style CSS
st.markdown("""
<style>
    .stApp {
        background: #1a1a1a !important;
        font-family: 'Segoe UI', -apple-system, sans-serif;
    }
    
    .header {
        background: linear-gradient(135deg, #228B22 0%, #1a5f1a 100%);
        color: white;
        padding: 20px 30px;
        margin: -1rem -1rem 2rem -1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    .alert-box {
        background: #FF0000;
        color: white;
        padding: 15px 20px;
        border-radius: 8px;
        margin: 15px 0;
        box-shadow: 0 4px 8px rgba(255,0,0,0.3);
        border-left: 5px solid #FFD700;
        font-weight: 600;
    }
    
    .metric-overlay {
        background: rgba(34, 139, 34, 0.9);
        color: white;
        padding: 10px 15px;
        border-radius: 6px;
        display: inline-block;
        margin: 5px;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def numpy_to_streamlit(img: np.ndarray) -> np.ndarray:
    """Convert BGR to RGB."""
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def draw_ai_overlay(image: np.ndarray, analysis_result: dict = None, alerts: list = None) -> np.ndarray:
    """
    Draw AI monitoring overlays on image - bounding boxes, alerts, metrics.
    Inspired by industrial AI monitoring systems.
    """
    overlay = image.copy()
    h, w = overlay.shape[:2]
    
    # Draw containers with bounding boxes (green)
    if analysis_result and 'containers' in analysis_result:
        for container in analysis_result.get('containers', []):
            if 'bbox' in container:
                bbox = np.array(container['bbox'], dtype=np.int32)
                # Draw green bounding box
                cv2.drawContours(overlay, [bbox], -1, (0, 255, 0), 3)
                
                # Add label
                cx, cy = container.get('centroid', (0, 0))
                label = f"Container: {container.get('type', 'unknown')}"
                cv2.putText(overlay, label, (cx - 80, cy - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Draw alert boxes (red warning boxes)
    if alerts:
        y_offset = 50
        for alert in alerts:
            # Draw red alert box
            box_height = 60
            box_width = 400
            cv2.rectangle(overlay, (20, y_offset), (20 + box_width, y_offset + box_height),
                         (0, 0, 255), -1)
            cv2.rectangle(overlay, (20, y_offset), (20 + box_width, y_offset + box_height),
                         (255, 215, 0), 3)
            
            # Alert text
            cv2.putText(overlay, "!", (35, y_offset + 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            cv2.putText(overlay, alert, (70, y_offset + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            y_offset += box_height + 10
    
    # Draw metrics overlay (green boxes in corner)
    if analysis_result:
        metrics = []
        if 'volume' in analysis_result:
            vol = analysis_result['volume'].get('volume_m3', 0)
            metrics.append(f"VOLUME: {vol:.0f} m³")
        if 'volume' in analysis_result:
            tons = analysis_result['volume'].get('tonnage_estimate', 0)
            metrics.append(f"TONNAGE: {tons:.1f} tons")
        
        y_pos = h - 100
        for metric in metrics:
            (text_width, text_height), _ = cv2.getTextSize(metric, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(overlay, (w - text_width - 30, y_pos - 25),
                         (w - 10, y_pos + 5), (34, 139, 34), -1)
            cv2.putText(overlay, metric, (w - text_width - 20, y_pos),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
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
    images = db.get_all_images()
    latest_img = None
    for img in images:
        if img['date'].startswith(selected_end.isoformat()[:10]):
            latest_img = img
            break
    
    # Collect alerts
    alerts = []
    if latest_img and latest_img.get('materials_json'):
        materials = json.loads(latest_img['materials_json'])
        contamination = materials.get('mixed', 0)
        if contamination > 20:
            alerts.append(f"HIGH CONTAMINATION: {contamination:.1f}%")
        
        purity = calculate_purity_score(materials)
        if purity < 70:
            alerts.append(f"LOW PURITY SCORE: {purity:.0f}%")
    
    # Show alerts
    if alerts:
        for alert in alerts:
            st.markdown(f'<div class="alert-box">⚠️ {alert}</div>', unsafe_allow_html=True)
    
    # Main Image with AI Overlays
    st.markdown("### 🎯 Live Site Monitoring")
    
    images = find_timelapse_images()
    images_in_range = [
        (img_path, dt) for img_path, dt in images
        if selected_start <= dt.date() <= selected_end
    ]
    
    if images_in_range:
        # Get latest image
        img_path, img_date = images_in_range[-1]
        image = cv2.imread(str(img_path))
        
        if image is not None:
            # Try to get analysis result
            analysis_result = None
            try:
                pipeline = DebrisAnalysisPipeline()
                analysis_result = pipeline.analyze(image)
            except:
                # Fallback: create basic result from database
                if latest_img:
                    analysis_result = {
                        'volume': {
                            'volume_m3': latest_img.get('volume_m3', 0),
                            'tonnage_estimate': latest_img.get('tonnage_estimate', 0),
                        },
                        'containers': [],  # Would need to query containers table
                    }
            
            # Draw AI overlays
            overlay_image = draw_ai_overlay(image, analysis_result, alerts)
            
            col_img1, col_img2 = st.columns([2, 1])
            
            with col_img1:
                st.image(numpy_to_streamlit(overlay_image), caption=f"AI-Enhanced View - {img_date.strftime('%B %Y')}", use_container_width=True)
            
            with col_img2:
                st.markdown("#### 📊 Real-Time Metrics")
                
                st.metric("Containers", latest.get('containers', 0))
                st.metric("Tonnage", f"{latest.get('tonnage', 0):.1f} tons")
                st.metric("Volume", f"{latest.get('volume_m3', 0):.0f} m³")
                
                if latest_img and latest_img.get('materials_json'):
                    materials = json.loads(latest_img['materials_json'])
                    purity = calculate_purity_score(materials)
                    st.metric("Purity", f"{purity:.0f}%")
    
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
            contamination = materials.get('mixed', 0)
            
            st.metric("Purity Score", f"{purity:.0f}%")
            if contamination > 20:
                st.error(f"⚠️ High contamination: {contamination:.1f}%")
            else:
                st.success(f"✓ Clean: {contamination:.1f}% mixed")
    
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
            st.metric("Throughput", f"{throughput:.1f} tons/day")
        
        with col_e2:
            arrivals = changes['summary'].get('arrivals', 0) if changes.get('summary') else 0
            st.metric("Arrivals", arrivals)
        
        with col_e3:
            departures = changes['summary'].get('departures', 0) if changes.get('summary') else 0
            st.metric("Departures", departures)
        
        with col_e4:
            net = changes['summary'].get('net_change', 0) if changes.get('summary') else 0
            st.metric("Net Change", net, delta=net)
    
    # Trend Chart
    st.markdown("---")
    st.markdown("### 📈 Volume Trend Analysis")
    
    if timeline.get('values') and len(timeline['values']) >= 2:
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            current = timeline['values'][-1]
            st.metric("Current Volume", f"{current:.0f} m³")
        with col_t2:
            recent_trend = timeline['values'][-1] - timeline['values'][-2]
            st.metric("Recent Change", f"{recent_trend:.0f} m³", delta=recent_trend if abs(recent_trend) > 100 else None)
        with col_t3:
            if len(timeline['values']) >= 2:
                first_vol = timeline['values'][0]
                last_vol = timeline['values'][-1]
                total_change = ((last_vol - first_vol) / first_vol * 100) if first_vol > 0 else 0
                st.metric("Total Change", f"{total_change:+.1f}%", delta=total_change)
        
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
            paper_bgcolor='#1a1a1a',
            plot_bgcolor='#2a2a2a',
            font=dict(color='white'),
            showlegend=False,
            xaxis=dict(gridcolor='#444'),
            yaxis=dict(gridcolor='#444', title='Volume (m³)'),
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


if __name__ == "__main__":
    main()
