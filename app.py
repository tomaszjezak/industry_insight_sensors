"""
EDCO Insights Dashboard - Industrial Modern Design
Professional dashboard for recycling yard operations.
"""

import streamlit as st
import numpy as np
from pathlib import Path
import cv2
from datetime import datetime, date
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Page config
st.set_page_config(
    page_title="EDCO Insights",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import components
from src.scraper import list_available_images
from src.pipeline import DebrisAnalysisPipeline
from src.query_engine import QueryEngine
from src.llm_query import LLMQueryInterface
from src.timelapse import find_timelapse_images, get_timelapse_summary, extract_date_from_filename
from src.timeseries_db import TimeseriesDB


# Custom CSS - Industrial Modern Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Montserrat:wght@700;800&display=swap');
    
    /* Reset and base styles */
    .stApp {
        background: #F5F5F5 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Header/Navigation */
    .main-header {
        background: linear-gradient(135deg, #228B22 0%, #32A632 100%);
        padding: 15px 30px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        position: sticky;
        top: 0;
        z-index: 100;
    }
    
    .header-content {
        max-width: 1400px;
        margin: 0 auto;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .logo {
        font-family: 'Montserrat', sans-serif;
        font-size: 1.8em;
        font-weight: 800;
        color: white;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .header-actions {
        display: flex;
        gap: 15px;
        align-items: center;
    }
    
    /* Hero Section */
    .hero-section {
        background: linear-gradient(rgba(34, 139, 34, 0.9), rgba(34, 139, 34, 0.7)),
                    url('https://images.unsplash.com/photo-1517089596392-fb9a9033e05b?w=1200') center/cover;
        padding: 60px 30px;
        color: white;
        margin-bottom: 30px;
    }
    
    .hero-content {
        max-width: 1400px;
        margin: 0 auto;
    }
    
    .site-name {
        font-family: 'Montserrat', sans-serif;
        font-size: 2.5em;
        font-weight: 800;
        margin-bottom: 10px;
    }
    
    .hero-stats {
        display: flex;
        gap: 40px;
        margin-top: 30px;
    }
    
    .hero-stat {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        padding: 20px 30px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .hero-stat-value {
        font-size: 2.5em;
        font-weight: 700;
        font-family: 'Montserrat', sans-serif;
    }
    
    .hero-stat-label {
        font-size: 0.9em;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Section Headers */
    .section-header {
        font-family: 'Montserrat', sans-serif;
        font-size: 2em;
        font-weight: 700;
        color: #333;
        margin: 40px 0 20px 0;
        text-align: center;
    }
    
    /* Insight Cards */
    .insight-card {
        background: white;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 4px solid #228B22;
        transition: transform 0.2s, box-shadow 0.2s;
        height: 100%;
    }
    
    .insight-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.15);
    }
    
    .insight-card.alert {
        border-left-color: #FF4500;
    }
    
    .insight-title {
        font-weight: 600;
        color: #333;
        font-size: 1.1em;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .insight-value {
        font-family: 'Montserrat', sans-serif;
        font-size: 2.2em;
        font-weight: 700;
        color: #228B22;
        margin: 10px 0;
    }
    
    .insight-value.alert {
        color: #FF4500;
    }
    
    .insight-trend {
        font-size: 0.9em;
        color: #666;
        display: flex;
        align-items: center;
        gap: 5px;
    }
    
    /* Data Section */
    .data-section {
        background: white;
        border-radius: 12px;
        padding: 30px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 30px;
    }
    
    /* Date Range Picker Styling */
    .stDateInput > div > div {
        background: white;
        border-radius: 8px;
    }
    
    /* Metrics Grid */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin: 20px 0;
    }
    
    .metric-box {
        background: white;
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        border: 2px solid #EEE;
    }
    
    .metric-value {
        font-family: 'Montserrat', sans-serif;
        font-size: 2em;
        font-weight: 700;
        color: #228B22;
        margin: 10px 0;
    }
    
    .metric-label {
        color: #666;
        font-size: 0.9em;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Image overlay styles */
    .image-container {
        position: relative;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Alert badges */
    .alert-badge {
        background: #FF4500;
        color: white;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 600;
    }
    
    /* Table styling */
    .data-table {
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)


def numpy_to_streamlit(img: np.ndarray) -> np.ndarray:
    """Convert BGR to RGB for Streamlit."""
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def format_currency(value: float) -> str:
    """Format currency."""
    if value >= 1000000:
        return f"${value/1000000:.2f}M"
    elif value >= 1000:
        return f"${value/1000:.1f}K"
    return f"${value:,.0f}"


def format_large_number(value: float) -> str:
    """Format large numbers."""
    if value >= 1000000:
        return f"{value/1000000:.2f}M"
    elif value >= 1000:
        return f"{value/1000:.1f}K"
    return f"{value:,.0f}"


def main():
    # Header/Navigation
    st.markdown("""
    <div class="main-header">
        <div class="header-content">
            <div class="logo">
                ♻️ EDCO Insights
            </div>
            <div class="header-actions">
                <button style="background: rgba(255,255,255,0.2); color: white; border: 1px solid rgba(255,255,255,0.3); padding: 8px 15px; border-radius: 5px; cursor: pointer;">
                    🔔 Notifications
                </button>
                <button style="background: rgba(255,255,255,0.2); color: white; border: 1px solid rgba(255,255,255,0.3); padding: 8px 15px; border-radius: 5px; cursor: pointer;">
                    ⚙️ Settings
                </button>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Hero/Dashboard Overview
    timelapse_summary = get_timelapse_summary()
    engine = QueryEngine()
    
    # Get current stats (latest image)
    current_stats = {}
    if timelapse_summary['total_images'] > 0:
        timeline = engine.query('timeline')
        if timeline.get('timeline'):
            latest = timeline['timeline'][-1]
            current_stats = {
                'volume': latest.get('volume_m3', 0),
                'tonnage': latest.get('tonnage', 0),
                'containers': latest.get('containers', 0),
                'date': latest.get('date', ''),
            }
    
    st.markdown(f"""
    <div class="hero-section">
        <div class="hero-content">
            <div class="site-name">Site: 38782 - EDCO Recycling Member</div>
            <div style="font-size: 1.1em; opacity: 0.95;">Real-time visibility into recycling operations</div>
            <div class="hero-stats">
                <div class="hero-stat">
                    <div class="hero-stat-value">{format_large_number(current_stats.get('volume', 0))}</div>
                    <div class="hero-stat-label">Current Volume (m³)</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-value">{format_large_number(current_stats.get('tonnage', 0))}</div>
                    <div class="hero-stat-label">Tonnage (tons)</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-value">{current_stats.get('containers', 0)}</div>
                    <div class="hero-stat-label">Active Containers</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main Content Container
    main_container = st.container()
    
    with main_container:
        # Date Range Selector (Top Right)
        col_date1, col_date2 = st.columns([3, 1])
        
        with col_date2:
            st.markdown("<br>", unsafe_allow_html=True)
            if timelapse_summary['total_images'] > 0:
                if timelapse_summary['date_range']:
                    start_date = datetime.fromisoformat(timelapse_summary['date_range']['start']).date()
                    end_date = datetime.fromisoformat(timelapse_summary['date_range']['end']).date()
                else:
                    start_date = date(2015, 1, 1)
                    end_date = date.today()
                
                date_range = st.date_input(
                    "📅 Date Range",
                    value=(start_date, end_date),
                    min_value=start_date,
                    max_value=end_date,
                    key="date_range_selector",
                )
                
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    selected_start, selected_end = date_range
                elif isinstance(date_range, date):
                    selected_start = date_range
                    selected_end = date_range
                else:
                    selected_start = start_date
                    selected_end = end_date
            else:
                selected_start = date(2015, 1, 1)
                selected_end = date.today()
        
        # Business Insights Section
        st.markdown('<div class="section-header">Business Insights</div>', unsafe_allow_html=True)
        
        if timelapse_summary['total_images'] > 0:
            # Get insights for date range
            inventory = engine.query('inventory', date_range=(selected_start.isoformat(), selected_end.isoformat()))
            changes = engine.query('changes', date_range=(selected_start.isoformat(), selected_end.isoformat()))
            revenue = engine.query('revenue', date_range=(selected_start.isoformat(), selected_end.isoformat()))
            timeline = engine.query('timeline', date_range=(selected_start.isoformat(), selected_end.isoformat()))
            
            # Insight Cards Grid
            col_i1, col_i2, col_i3, col_i4 = st.columns(4)
            
            with col_i1:
                # Volume Trends Card
                st.markdown("""
                <div class="insight-card">
                    <div class="insight-title">📊 Volume Trends</div>
                    <div class="insight-value" id="volume-value">-</div>
                    <div class="insight-trend">Average over selected period</div>
                </div>
                """, unsafe_allow_html=True)
                
                if inventory.get('snapshots'):
                    avg_vol = sum(s.get('volume_m3', 0) for s in inventory['snapshots']) / len(inventory['snapshots'])
                    max_vol = max(s.get('volume_m3', 0) for s in inventory['snapshots'])
                    min_vol = min(s.get('volume_m3', 0) for s in inventory['snapshots'])
                    
                    # Calculate trend
                    if len(inventory['snapshots']) >= 2:
                        first_vol = inventory['snapshots'][0].get('volume_m3', 0)
                        last_vol = inventory['snapshots'][-1].get('volume_m3', 0)
                        trend_pct = ((last_vol - first_vol) / first_vol * 100) if first_vol > 0 else 0
                        trend_icon = "📈" if trend_pct > 0 else "📉"
                        trend_color = "#228B22" if trend_pct > 0 else "#FF4500"
                    else:
                        trend_pct = 0
                        trend_icon = "➡️"
                        trend_color = "#666"
                    
                    st.markdown(f"""
                    <script>
                    document.getElementById('volume-value').innerHTML = '{format_large_number(avg_vol)}';
                    document.getElementById('volume-value').parentElement.innerHTML += 
                        '<div class="insight-trend" style="color: {trend_color};">{trend_icon} {trend_pct:+.1f}% vs start</div>';
                    </script>
                    """, unsafe_allow_html=True)
            
            with col_i2:
                # Safety & Efficiency Card
                st.markdown("""
                <div class="insight-card">
                    <div class="insight-title">⚠️ Safety & Efficiency</div>
                    <div class="insight-value" id="safety-value">-</div>
                    <div class="insight-trend">Active alerts</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Calculate efficiency metrics
                if changes.get('summary'):
                    total_changes = changes['summary'].get('total_changes', 0)
                    efficiency_score = max(0, 100 - (total_changes * 5))  # Simple heuristic
                    st.markdown(f"""
                    <script>
                    document.getElementById('safety-value').innerHTML = '{efficiency_score:.0f}%';
                    document.getElementById('safety-value').parentElement.innerHTML += 
                        '<div class="insight-trend">Operations efficiency score</div>';
                    </script>
                    """, unsafe_allow_html=True)
            
            with col_i3:
                # Inventory & Equipment Card
                st.markdown("""
                <div class="insight-card">
                    <div class="insight-title">📦 Inventory & Equipment</div>
                    <div class="insight-value" id="inventory-value">-</div>
                    <div class="insight-trend">Current status</div>
                </div>
                """, unsafe_allow_html=True)
                
                if inventory.get('snapshots'):
                    latest = inventory['snapshots'][-1]
                    container_count = latest.get('containers', 0)
                    utilization = min(100, (latest.get('volume_m3', 0) / 100000 * 100)) if latest.get('volume_m3', 0) > 0 else 0
                    
                    status_color = "#228B22" if utilization < 70 else "#FF4500"
                    status_text = "Optimal" if utilization < 70 else "High Capacity"
                    
                    st.markdown(f"""
                    <script>
                    document.getElementById('inventory-value').innerHTML = '{container_count}';
                    document.getElementById('inventory-value').parentElement.innerHTML += 
                        '<div class="insight-trend" style="color: {status_color};">{status_text} ({utilization:.0f}% utilized)</div>';
                    </script>
                    """, unsafe_allow_html=True)
            
            with col_i4:
                # Revenue Potential Card
                st.markdown("""
                <div class="insight-card">
                    <div class="insight-title">💰 Revenue Potential</div>
                    <div class="insight-value" id="revenue-value">-</div>
                    <div class="insight-trend">Estimated value</div>
                </div>
                """, unsafe_allow_html=True)
                
                if revenue.get('total_value'):
                    rev_value = revenue['total_value']
                    st.markdown(f"""
                    <script>
                    document.getElementById('revenue-value').innerHTML = '{format_currency(rev_value)}';
                    document.getElementById('revenue-value').parentElement.innerHTML += 
                        '<div class="insight-trend">Based on material classification</div>';
                    </script>
                    """, unsafe_allow_html=True)
            
            # Site Data & Photos Section
            st.markdown('<div class="section-header">Site Data & Photos</div>', unsafe_allow_html=True)
            
            # Date Range Display
            st.markdown(f"""
            <div style="text-align: center; color: #666; margin-bottom: 20px;">
                Analyzing data from <strong>{selected_start.strftime('%B %d, %Y')}</strong> to 
                <strong>{selected_end.strftime('%B %d, %Y')}</strong>
            </div>
            """, unsafe_allow_html=True)
            
            # Image Comparison
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
                        st.image(numpy_to_streamlit(img1), use_container_width=True)
                
                with col_img2:
                    if len(images_in_range) > 1:
                        img2_path, date2 = images_in_range[-1]
                        img2 = cv2.imread(str(img2_path))
                        if img2 is not None:
                            st.markdown(f"**{date2.strftime('%B %Y')}**")
                            st.image(numpy_to_streamlit(img2), use_container_width=True)
                    else:
                        st.info("Only one image in selected range")
            
            # Volume Over Time Chart
            if timeline.get('values'):
                st.markdown("### Volume Over Time")
                
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
                    plot_bgcolor='#FAFAFA',
                    font=dict(family='Inter', size=12),
                    xaxis=dict(title='Date', gridcolor='#E0E0E0'),
                    yaxis=dict(title='Volume (m³)', gridcolor='#E0E0E0'),
                    hovermode='x unified',
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed Metrics Table
            if inventory.get('snapshots'):
                st.markdown("### Detailed Metrics")
                
                df_data = []
                for snap in inventory['snapshots']:
                    df_data.append({
                        'Date': datetime.fromisoformat(snap['date']).strftime('%Y-%m-%d'),
                        'Volume (m³)': f"{snap.get('volume_m3', 0):.1f}",
                        'Area (m²)': f"{snap.get('area_m2', 0):.1f}",
                        'Tonnage (tons)': f"{snap.get('tonnage', 0):.1f}",
                        'Containers': snap.get('containers', 0),
                        'Dominant Material': snap.get('dominant_material', 'unknown').capitalize(),
                    })
                
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            
            # AI-Powered Analysis Section
            st.markdown('<div class="section-header">AI-Powered Analysis</div>', unsafe_allow_html=True)
            
            col_ai1, col_ai2 = st.columns(2)
            
            with col_ai1:
                st.markdown("""
                <div class="insight-card">
                    <div class="insight-title">🔄 Material Flow</div>
                    <div style="margin-top: 15px;">
                """, unsafe_allow_html=True)
                
                if changes.get('summary'):
                    st.metric("Total Arrivals", changes['summary'].get('arrivals', 0))
                    st.metric("Total Departures", changes['summary'].get('departures', 0))
                    st.metric("Net Change", changes['summary'].get('net_change', 0))
                
                st.markdown("</div></div>", unsafe_allow_html=True)
            
            with col_ai2:
                st.markdown("""
                <div class="insight-card">
                    <div class="insight-title">📈 Process Optimization</div>
                    <div style="margin-top: 15px; color: #666;">
                """, unsafe_allow_html=True)
                
                if timeline.get('timeline') and len(timeline['timeline']) >= 2:
                    # Calculate throughput
                    first = timeline['timeline'][0]
                    last = timeline['timeline'][-1]
                    days = (datetime.fromisoformat(last['date']) - datetime.fromisoformat(first['date'])).days
                    if days > 0:
                        throughput = (last.get('tonnage', 0) - first.get('tonnage', 0)) / days
                        st.metric("Avg Throughput", f"{throughput:.1f} tons/day")
                
                st.markdown("</div></div>", unsafe_allow_html=True)
            
            # Reports Section
            st.markdown('<div class="section-header">Generate Reports</div>', unsafe_allow_html=True)
            
            with st.expander("📄 Export Custom Report", expanded=False):
                col_r1, col_r2 = st.columns(2)
                
                with col_r1:
                    report_metrics = st.multiselect(
                        "Select Metrics",
                        ["Volume", "Tonnage", "Area", "Containers", "Materials", "Revenue"],
                        default=["Volume", "Tonnage", "Revenue"]
                    )
                    report_format = st.selectbox("Format", ["PDF", "Excel", "CSV"])
                
                with col_r2:
                    st.markdown("**Report Preview**")
                    st.info(f"Will include: {', '.join(report_metrics)}")
                    st.info(f"Date range: {selected_start.strftime('%Y-%m-%d')} to {selected_end.strftime('%Y-%m-%d')}")
                
                if st.button("📥 Download Report", type="primary"):
                    st.success("Report generation started! (Demo mode)")
        
        else:
            st.info("No timelapse data available. Process images to see insights.")
            st.code("python -m src.batch_processor", language="bash")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px; font-size: 0.9em;">
        <p><strong>© 2026 EDCO Insights</strong> - Powered by AI for Recycling Efficiency</p>
        <p style="margin-top: 10px;">
            Data sourced from existing cameras; insights for demonstration purposes.<br>
            Need deeper analysis? <a href="mailto:tomaszjezak@ucsd.edu" style="color: #228B22;">Contact us</a>
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
