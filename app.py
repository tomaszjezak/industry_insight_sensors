"""
EDCO Insights Dashboard - Single-Page, No-Scroll Design
Professional recycling yard operations dashboard.
"""

import streamlit as st
import numpy as np
from pathlib import Path
import cv2
from datetime import datetime, date, timedelta
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json

# Page config
st.set_page_config(
    page_title="EDCO Insights",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Import components
from src.scraper import list_available_images
from src.query_engine import QueryEngine
from src.timelapse import find_timelapse_images, get_timelapse_summary
from src.timeseries_db import TimeseriesDB


# Custom CSS - Single Page, No Scroll
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Montserrat:wght@700;800&display=swap');
    
    /* Prevent scrolling */
    .stApp {
        background: #F5F5F5 !important;
        font-family: 'Inter', sans-serif !important;
        overflow: hidden !important;
        height: 100vh !important;
    }
    
    .main-container {
        height: 100vh;
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }
    
    /* Compact Header */
    .main-header {
        background: linear-gradient(135deg, #228B22 0%, #32A632 100%);
        padding: 10px 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        flex-shrink: 0;
    }
    
    .logo {
        font-family: 'Montserrat', sans-serif;
        font-size: 1.5em;
        font-weight: 800;
        color: white;
    }
    
    /* Date Slider Container */
    .date-slider-container {
        background: white;
        padding: 15px 20px;
        border-radius: 8px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Content Area - Grid Layout */
    .content-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        grid-template-rows: auto auto auto;
        gap: 10px;
        padding: 10px;
        flex: 1;
        overflow-y: auto;
        max-height: calc(100vh - 200px);
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #228B22;
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    
    .metric-card.alert {
        border-left-color: #FF4500;
    }
    
    .metric-card.warning {
        border-left-color: #FFA500;
    }
    
    .metric-title {
        font-size: 0.85em;
        font-weight: 600;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-family: 'Montserrat', sans-serif;
        font-size: 1.8em;
        font-weight: 700;
        color: #228B22;
        margin: 5px 0;
    }
    
    .metric-value.alert {
        color: #FF4500;
    }
    
    .metric-subtitle {
        font-size: 0.75em;
        color: #999;
        margin-top: auto;
    }
    
    /* Image Section */
    .image-section {
        grid-column: 1 / -1;
        background: white;
        border-radius: 8px;
        padding: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        display: flex;
        gap: 8px;
        max-height: 150px;
        overflow: hidden;
    }
    
    .image-wrapper {
        flex: 1;
        border-radius: 4px;
        overflow: hidden;
    }
    
    /* Chart Section */
    .chart-section {
        grid-column: 1 / -1;
        background: white;
        border-radius: 8px;
        padding: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        max-height: 180px;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Compact styling */
    .stMetric {
        padding: 5px 0;
    }
    
    .stMetric > div {
        padding: 0;
    }
</style>
""", unsafe_allow_html=True)


def numpy_to_streamlit(img: np.ndarray) -> np.ndarray:
    """Convert BGR to RGB for Streamlit."""
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def date_to_timestamp(d: date) -> float:
    """Convert date to timestamp for slider."""
    return datetime.combine(d, datetime.min.time()).timestamp()


def timestamp_to_date(ts: float) -> date:
    """Convert timestamp to date."""
    return datetime.fromtimestamp(ts).date()


def create_date_slider(start_date: date, end_date: date, key_prefix: str = "date"):
    """Create dual-pointer date range slider using date_input."""
    col1, col2 = st.columns(2)
    
    with col1:
        start_date_selected = st.date_input(
            "Start Date",
            value=start_date,
            min_value=start_date,
            max_value=end_date,
            key=f"{key_prefix}_start"
        )
    
    with col2:
        end_date_selected = st.date_input(
            "End Date",
            value=end_date,
            min_value=start_date_selected if isinstance(start_date_selected, date) else start_date,
            max_value=end_date,
            key=f"{key_prefix}_end"
        )
    
    # Handle tuple from date_input
    if isinstance(start_date_selected, tuple):
        start_date_selected = start_date_selected[0]
    if isinstance(end_date_selected, tuple):
        end_date_selected = end_date_selected[0]
    
    # Ensure start <= end
    if start_date_selected > end_date_selected:
        start_date_selected = end_date_selected
    
    return start_date_selected, end_date_selected


def calculate_storage_utilization(current_volume: float, historical_volumes: list) -> float:
    """Calculate storage utilization %."""
    if not historical_volumes:
        return 0
    max_volume = max(historical_volumes) * 1.2  # 20% buffer
    if max_volume == 0:
        return 0
    return min(100, (current_volume / max_volume) * 100)


def calculate_purity_score(materials: dict) -> float:
    """Calculate material purity score (0-100)."""
    if not materials:
        return 0
    # Purity = 100 - mixed_percentage, or dominant material percentage
    mixed_pct = materials.get('mixed', 0)
    dominant_pct = max(materials.values()) if materials else 0
    return max(dominant_pct, 100 - mixed_pct)


def calculate_throughput(start_tonnage: float, end_tonnage: float, days: int) -> float:
    """Calculate throughput rate (tons/day)."""
    if days <= 0:
        return 0
    return (end_tonnage - start_tonnage) / days


def calculate_housekeeping_score(volume: float, area: float) -> float:
    """Calculate housekeeping score (heuristic based on volume/area ratio)."""
    if area == 0:
        return 0
    # Higher volume/area ratio = better organized (piled up, not scattered)
    ratio = volume / area
    # Normalize to 0-100 (heuristic thresholds)
    if ratio > 10:  # Well piled
        return 95
    elif ratio > 5:
        return 85
    elif ratio > 2:
        return 70
    else:  # Scattered
        return 50


def simple_forecast(values: list, days_ahead: int = 7) -> float:
    """Simple linear forecast for next N days."""
    if len(values) < 2:
        return values[-1] if values else 0
    
    # Linear regression on last 3 points
    n = min(3, len(values))
    recent = values[-n:]
    x = list(range(n))
    
    # Simple linear fit
    x_mean = sum(x) / n
    y_mean = sum(recent) / n
    
    numerator = sum((x[i] - x_mean) * (recent[i] - y_mean) for i in range(n))
    denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
    
    if denominator == 0:
        return recent[-1]
    
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    
    # Forecast
    forecast_x = n + days_ahead
    forecast_y = slope * forecast_x + intercept
    
    return max(0, forecast_y)


def main():
    # Initialize
    timelapse_summary = get_timelapse_summary()
    engine = QueryEngine()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <div class="logo">♻️ EDCO Insights - Site 38782</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Date Range Slider
    if timelapse_summary['total_images'] > 0:
        if timelapse_summary['date_range']:
            start_date = datetime.fromisoformat(timelapse_summary['date_range']['start']).date()
            end_date = datetime.fromisoformat(timelapse_summary['date_range']['end']).date()
        else:
            start_date = date(2015, 1, 1)
            end_date = date.today()
        
        st.markdown('<div class="date-slider-container">', unsafe_allow_html=True)
        st.markdown("**📅 Select Date Range**")
        selected_start, selected_end = create_date_slider(start_date, end_date)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Query data for selected range
        inventory = engine.query('inventory', date_range=(selected_start.isoformat(), selected_end.isoformat()))
        changes = engine.query('changes', date_range=(selected_start.isoformat(), selected_end.isoformat()))
        revenue = engine.query('revenue', date_range=(selected_start.isoformat(), selected_end.isoformat()))
        timeline = engine.query('timeline', date_range=(selected_start.isoformat(), selected_end.isoformat()))
        
        # Get snapshots
        snapshots = inventory.get('snapshots', [])
        timeline_data = timeline.get('timeline', [])
        
        # Main Content Grid
        st.markdown('<div class="content-grid">', unsafe_allow_html=True)
        
        # A. Inventory & Volume Tracking (Top Left)
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">📦 Inventory & Volume</div>
        </div>
        """, unsafe_allow_html=True)
        
        if snapshots:
                latest = snapshots[-1]
                all_volumes = [s.get('volume_m3', 0) for s in snapshots]
                current_volume = latest.get('volume_m3', 0)
                current_tonnage = latest.get('tonnage', 0)
                container_count = latest.get('containers', 0)
                
                # Storage utilization
                utilization = calculate_storage_utilization(current_volume, all_volumes)
                
                # Volume trend
                if len(snapshots) >= 2:
                    first_vol = snapshots[0].get('volume_m3', 0)
                    last_vol = snapshots[-1].get('volume_m3', 0)
                    trend_pct = ((last_vol - first_vol) / first_vol * 100) if first_vol > 0 else 0
                    trend_icon = "📈" if trend_pct > 0 else "📉"
                else:
                    trend_pct = 0
                    trend_icon = "➡️"
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Containers", f"{container_count}")
                    st.metric("Tonnage", f"{current_tonnage:.1f} tons")
                with col2:
                    st.metric("Storage", f"{utilization:.0f}%", delta=trend_pct)
                    st.metric("Volume", f"{current_volume:.0f} m³")
        
        # B. Material Quality & Contamination (Top Right)
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">🧱 Material Quality</div>
        </div>
        """, unsafe_allow_html=True)
        
        if snapshots:
                latest = snapshots[-1]
                # Get materials from database
                db = TimeseriesDB()
                images = db.get_all_images()
                latest_img = None
                for img in images:
                    if img['date'].startswith(selected_end.isoformat()[:10]):
                        latest_img = img
                        break
                
                if latest_img and latest_img.get('materials_json'):
                    materials = json.loads(latest_img['materials_json'])
                    purity = calculate_purity_score(materials)
                    contamination = materials.get('mixed', 0)
                    
                    # Show composition
                    st.markdown("**Composition:**")
                    for material, pct in materials.items():
                        st.progress(pct / 100, text=f"{material.capitalize()}: {pct:.1f}%")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        alert_class = "alert" if purity < 70 else "warning" if purity < 85 else ""
                        # Calculate purity delta (vs 85% target)
                        purity_delta = purity - 85
                        st.metric("Purity Score", f"{purity:.0f}%", delta=purity_delta if abs(purity_delta) > 5 else None)
                    with col2:
                        if contamination > 20:
                            st.error(f"⚠️ High contamination: {contamination:.1f}%")
                        else:
                            st.success(f"✓ Clean: {contamination:.1f}% mixed")
                else:
                    st.info("Material data not available")
        
        # C. Operational Efficiency & Throughput (Middle Left)
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">⚙️ Operational Efficiency</div>
        </div>
        """, unsafe_allow_html=True)
        
        if timeline_data and len(timeline_data) >= 2:
            first = timeline_data[0]
            last = timeline_data[-1]
            days = (datetime.fromisoformat(last['date']) - datetime.fromisoformat(first['date'])).days
            
            throughput = calculate_throughput(
                first.get('tonnage', 0),
                last.get('tonnage', 0),
                max(1, days)
            )
            
            if changes.get('summary'):
                arrivals = changes['summary'].get('arrivals', 0)
                departures = changes['summary'].get('departures', 0)
                net_change = changes['summary'].get('net_change', 0)
            else:
                arrivals = departures = net_change = 0
            
            st.metric("Throughput", f"{throughput:.1f} tons/day")
            st.metric("Arrivals", arrivals, delta=arrivals if arrivals > 0 else None)
            st.metric("Departures", departures, delta=-departures if departures > 0 else None)
            st.metric("Net Change", net_change, delta=net_change)
        else:
            st.info("Insufficient data for efficiency metrics")
        
        # D. Safety & Compliance Monitoring (Middle Right)
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">⚠️ Safety & Compliance</div>
        </div>
        """, unsafe_allow_html=True)
        
        if snapshots:
            latest = snapshots[-1]
            volume = latest.get('volume_m3', 0)
            area = latest.get('area_m2', 0)
            
            housekeeping = calculate_housekeeping_score(volume, area)
            
            # Compliance status
            if housekeeping >= 85:
                compliance_status = "✓ Compliant"
                compliance_color = "#228B22"
            elif housekeeping >= 70:
                compliance_status = "⚠️ Moderate"
                compliance_color = "#FFA500"
            else:
                compliance_status = "⚠️ Needs Attention"
                compliance_color = "#FF4500"
            
            st.metric("Housekeeping", f"{housekeeping:.0f}%")
            st.markdown(f'<div style="color: {compliance_color}; font-weight: 600;">{compliance_status}</div>', unsafe_allow_html=True)
            st.metric("Hazard Alerts", 0)
        else:
            st.info("Safety data not available")
        
        # Image Display (Center, Full Width)
        st.markdown('<div class="image-section">', unsafe_allow_html=True)
        
        images = find_timelapse_images()
        images_in_range = [
            (img_path, dt) for img_path, dt in images
            if selected_start <= dt.date() <= selected_end
        ]
        
        if images_in_range:
            col_img1, col_img2 = st.columns(2)
            
            with col_img1:
                img1_path, date1 = images_in_range[0]
                img1 = cv2.imread(str(img1_path))
                if img1 is not None:
                    st.markdown(f"**{date1.strftime('%B %Y')}**")
                    st.image(numpy_to_streamlit(img1), width='stretch')
            
            with col_img2:
                if len(images_in_range) > 1:
                    img2_path, date2 = images_in_range[-1]
                    img2 = cv2.imread(str(img2_path))
                    if img2 is not None:
                        st.markdown(f"**{date2.strftime('%B %Y')}**")
                        st.image(numpy_to_streamlit(img2), width='stretch')
                else:
                    st.info("Only one image in range")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # E. Predictive & Trend Analytics (Bottom, Full Width)
        st.markdown('<div class="chart-section">', unsafe_allow_html=True)
        st.markdown("**Predictive & Trend Analytics**")
        
        if timeline.get('values') and len(timeline['values']) >= 2:
            # Volume forecast
            forecast = simple_forecast(timeline['values'], 7)
            current = timeline['values'][-1]
            
            col_pred1, col_pred2, col_pred3 = st.columns(3)
            with col_pred1:
                # Calculate absolute forecast change
                forecast_delta = forecast - current
                st.metric("7-Day Forecast", f"{forecast:.0f} m³", delta=forecast_delta if abs(forecast_delta) > 100 else None)
            
            with col_pred2:
                # Diversion rate proxy (assume all detected material is recyclable)
                total_volume = sum(timeline['values'])
                diversion_rate = 100  # Heuristic: all detected = recyclable
                st.metric("Diversion Rate", f"{diversion_rate:.0f}%")
            
            with col_pred3:
                # Trend direction
                if len(timeline['values']) >= 2:
                    recent_trend = timeline['values'][-1] - timeline['values'][-2]
                    trend_value = f"{recent_trend:.0f} m³"
                    st.metric("Trend", trend_value, delta=recent_trend if abs(recent_trend) > 100 else None)
            
            # Trend Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=timeline['dates'],
                y=timeline['values'],
                mode='lines+markers',
                name='Volume',
                line=dict(color='#228B22', width=2),
                marker=dict(size=6, color='#228B22'),
            ))
            
            # Add trend line (simple linear)
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
                    line=dict(color='#FF4500', width=1, dash='dash'),
                ))
            
            fig.update_layout(
                height=150,
                paper_bgcolor='white',
                plot_bgcolor='#FAFAFA',
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(showgrid=True, gridcolor='#E0E0E0'),
                yaxis=dict(showgrid=True, gridcolor='#E0E0E0', title='Volume (m³)'),
            )
            
            st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)  # Close content-grid
        
    else:
        st.warning("No timelapse data available. Process images first.")
        st.code("python -m src.batch_processor", language="bash")


if __name__ == "__main__":
    main()
