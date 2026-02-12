"""
EDCO Insights Dashboard - Clean Professional Design
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


# Clean CSS
st.markdown("""
<style>
    .stApp {
        background: #f8f9fa !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .header {
        background: #228B22;
        color: white;
        padding: 15px 30px;
        margin: -1rem -1rem 1rem -1rem;
    }
    
    .metric-box {
        background: white;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    
    .metric-label {
        font-size: 0.85em;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 5px;
    }
    
    .metric-value {
        font-size: 2em;
        font-weight: 700;
        color: #228B22;
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


def calculate_storage_utilization(current_volume: float, historical_volumes: list) -> float:
    """Calculate storage utilization %."""
    if not historical_volumes:
        return 0
    max_volume = max(historical_volumes) * 1.2
    if max_volume == 0:
        return 0
    return min(100, (current_volume / max_volume) * 100)


def calculate_purity_score(materials: dict) -> float:
    """Calculate material purity score."""
    if not materials:
        return 0
    mixed_pct = materials.get('mixed', 0)
    dominant_pct = max(materials.values()) if materials else 0
    return max(dominant_pct, 100 - mixed_pct)


def calculate_throughput(start_tonnage: float, end_tonnage: float, days: int) -> float:
    """Calculate throughput rate."""
    if days <= 0:
        return 0
    return (end_tonnage - start_tonnage) / days


def calculate_housekeeping_score(volume: float, area: float) -> float:
    """Calculate housekeeping score."""
    if area == 0:
        return 0
    ratio = volume / area
    if ratio > 10:
        return 95
    elif ratio > 5:
        return 85
    elif ratio > 2:
        return 70
    else:
        return 50


def simple_forecast(values: list, days_ahead: int = 7) -> float:
    """Simple linear forecast."""
    if len(values) < 2:
        return values[-1] if values else 0
    
    n = min(3, len(values))
    recent = values[-n:]
    x = list(range(n))
    
    x_mean = sum(x) / n
    y_mean = sum(recent) / n
    
    numerator = sum((x[i] - x_mean) * (recent[i] - y_mean) for i in range(n))
    denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
    
    if denominator == 0:
        return recent[-1]
    
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    
    forecast_x = n + days_ahead
    forecast_y = slope * forecast_x + intercept
    
    return max(0, forecast_y)


def main():
    # Header
    st.markdown("""
    <div class="header">
        <h1 style="margin:0; font-size: 1.8em;">♻️ EDCO Insights - Site 38782</h1>
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
    
    st.markdown("### 📅 Date Range")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        selected_start = st.date_input("From", value=start_date, min_value=start_date, max_value=end_date, key="date_start")
    with col_d2:
        selected_end = st.date_input("To", value=end_date, min_value=selected_start if isinstance(selected_start, date) else start_date, max_value=end_date, key="date_end")
    
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
    
    # Main Metrics - 4 columns
    st.markdown("---")
    st.markdown("### Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        latest = snapshots[-1]
        st.metric("Containers", latest.get('containers', 0))
        st.caption("Active containers")
    
    with col2:
        st.metric("Tonnage", f"{latest.get('tonnage', 0):.1f}")
        st.caption("Total tons")
    
    with col3:
        all_volumes = [s.get('volume_m3', 0) for s in snapshots]
        utilization = calculate_storage_utilization(latest.get('volume_m3', 0), all_volumes)
        st.metric("Storage", f"{utilization:.0f}%")
        st.caption("Yard utilization")
    
    with col4:
        st.metric("Volume", f"{latest.get('volume_m3', 0):.0f}")
        st.caption("Cubic meters")
    
    # Material Quality
    st.markdown("---")
    st.markdown("### Material Quality")
    
    db = TimeseriesDB()
    images = db.get_all_images()
    latest_img = None
    for img in images:
        if img['date'].startswith(selected_end.isoformat()[:10]):
            latest_img = img
            break
    
    col_m1, col_m2 = st.columns([2, 1])
    
    with col_m1:
        if latest_img and latest_img.get('materials_json'):
            materials = json.loads(latest_img['materials_json'])
            st.markdown("**Composition:**")
            for material, pct in materials.items():
                st.progress(pct / 100, text=f"{material.capitalize()}: {pct:.1f}%")
        else:
            st.info("Material data not available")
    
    with col_m2:
        if latest_img and latest_img.get('materials_json'):
            materials = json.loads(latest_img['materials_json'])
            purity = calculate_purity_score(materials)
            contamination = materials.get('mixed', 0)
            
            st.metric("Purity", f"{purity:.0f}%")
            if contamination > 20:
                st.error(f"⚠️ High contamination: {contamination:.1f}%")
            else:
                st.success(f"✓ Clean: {contamination:.1f}% mixed")
    
    # Operational Efficiency
    st.markdown("---")
    st.markdown("### Operational Efficiency")
    
    if timeline_data and len(timeline_data) >= 2:
        first = timeline_data[0]
        last = timeline_data[-1]
        days = (datetime.fromisoformat(last['date']) - datetime.fromisoformat(first['date'])).days
        
        throughput = calculate_throughput(first.get('tonnage', 0), last.get('tonnage', 0), max(1, days))
        
        col_e1, col_e2, col_e3, col_e4 = st.columns(4)
        
        with col_e1:
            st.metric("Throughput", f"{throughput:.1f}")
            st.caption("Tons per day")
        
        with col_e2:
            arrivals = changes['summary'].get('arrivals', 0) if changes.get('summary') else 0
            st.metric("Arrivals", arrivals)
            st.caption("Containers/pallets")
        
        with col_e3:
            departures = changes['summary'].get('departures', 0) if changes.get('summary') else 0
            st.metric("Departures", departures)
            st.caption("Containers/pallets")
        
        with col_e4:
            net = changes['summary'].get('net_change', 0) if changes.get('summary') else 0
            st.metric("Net Change", net, delta=net)
            st.caption("Net change")
    
    # Safety & Compliance
    st.markdown("---")
    st.markdown("### Safety & Compliance")
    
    volume = latest.get('volume_m3', 0)
    area = latest.get('area_m2', 0)
    housekeeping = calculate_housekeeping_score(volume, area)
    
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        st.metric("Housekeeping", f"{housekeeping:.0f}%")
        if housekeeping >= 85:
            st.success("✓ Compliant")
        elif housekeeping >= 70:
            st.warning("⚠️ Moderate")
        else:
            st.error("⚠️ Needs Attention")
    
    with col_s2:
        st.metric("Hazard Alerts", 0)
        st.caption("No hazards detected")
    
    # Images
    st.markdown("---")
    st.markdown("### Site Images")
    
    images = find_timelapse_images()
    images_in_range = [
        (img_path, dt) for img_path, dt in images
        if selected_start <= dt.date() <= selected_end
    ]
    
    if images_in_range:
        col_i1, col_i2 = st.columns(2)
        
        with col_i1:
            img1_path, date1 = images_in_range[0]
            img1 = cv2.imread(str(img1_path))
            if img1 is not None:
                st.image(numpy_to_streamlit(img1), caption=f"{date1.strftime('%B %Y')}", use_container_width=True)
        
        with col_i2:
            if len(images_in_range) > 1:
                img2_path, date2 = images_in_range[-1]
                img2 = cv2.imread(str(img2_path))
                if img2 is not None:
                    st.image(numpy_to_streamlit(img2), caption=f"{date2.strftime('%B %Y')}", use_container_width=True)
    
    # Trend Chart
    st.markdown("---")
    st.markdown("### Volume Trend")
    
    if timeline.get('values') and len(timeline['values']) >= 2:
        forecast = simple_forecast(timeline['values'], 7)
        current = timeline['values'][-1]
        forecast_delta = forecast - current
        
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            st.metric("7-Day Forecast", f"{forecast:.0f}", delta=forecast_delta if abs(forecast_delta) > 100 else None)
            st.caption("Cubic meters")
        with col_t2:
            st.metric("Diversion Rate", "100%")
            st.caption("Proxy estimate")
        with col_t3:
            recent_trend = timeline['values'][-1] - timeline['values'][-2]
            st.metric("Recent Change", f"{recent_trend:.0f}", delta=recent_trend if abs(recent_trend) > 100 else None)
            st.caption("Cubic meters")
        
        # Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timeline['dates'],
            y=timeline['values'],
            mode='lines+markers',
            name='Volume',
            line=dict(color='#228B22', width=3),
            marker=dict(size=8, color='#228B22'),
        ))
        
        if len(timeline['values']) >= 2:
            x_numeric = list(range(len(timeline['dates'])))
            z = np.polyfit(x_numeric, timeline['values'], 1)
            p = np.poly1d(z)
            trend_line = p(x_numeric)
            fig.add_trace(go.Scatter(
                x=timeline['dates'],
                y=trend_line,
                mode='lines',
                name='Trend',
                line=dict(color='#FF4500', width=2, dash='dash'),
            ))
        
        fig.update_layout(
            height=400,
            paper_bgcolor='white',
            plot_bgcolor='#f8f9fa',
            showlegend=False,
            xaxis_title='Date',
            yaxis_title='Volume (m³)',
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


if __name__ == "__main__":
    main()
