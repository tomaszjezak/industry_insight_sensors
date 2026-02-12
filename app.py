"""
Enco Insights Dashboard - Clean, Simple UI
"""

import streamlit as st
import numpy as np
from pathlib import Path
import cv2
from datetime import datetime, date
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Enco Insights",
    page_icon="🏗️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Import components
from src.scraper import list_available_images
from src.pipeline import DebrisAnalysisPipeline
from src.query_engine import QueryEngine
from src.llm_query import LLMQueryInterface
from src.timelapse import find_timelapse_images, get_timelapse_summary, extract_date_from_filename
from src.timeseries_db import TimeseriesDB


# Custom CSS - Clean, Simple Design
st.markdown("""
<style>
    /* Main container */
    .main-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
        border: 2px solid #FF0000;
        border-radius: 8px;
        background: white;
    }
    
    /* Header */
    h1 {
        color: #FF0000 !important;
        font-weight: bold !important;
        font-size: 2em !important;
        margin-bottom: 20px;
        text-align: left;
    }
    
    /* Site info */
    .site-info {
        font-size: 1.2em;
        font-weight: bold;
        color: #333;
        margin-bottom: 10px;
    }
    
    /* Date selector */
    .date-selector {
        text-align: right;
        margin-bottom: 20px;
    }
    
    /* Image section */
    .image-section {
        margin: 20px 0;
        text-align: center;
    }
    
    /* Stats section */
    .stats-section {
        margin-top: 20px;
        padding: 15px;
        background: #f5f5f5;
        border-radius: 5px;
    }
    
    /* Remove Streamlit default styling */
    .stApp {
        background: white !important;
    }
    
    .stButton>button {
        background: #FF0000;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }
    
    .stButton>button:hover {
        background: #cc0000;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def numpy_to_streamlit(img: np.ndarray) -> np.ndarray:
    """Convert BGR to RGB for Streamlit."""
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    # Main Container
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Header
    st.markdown("# Enco Insights")
    
    # Top Row: Site Info + Dates
    col_top1, col_top2 = st.columns([1, 1])
    
    with col_top1:
        st.markdown('<div class="site-info">Site: 38782 Member</div>', unsafe_allow_html=True)
    
    with col_top2:
        st.markdown('<div class="date-selector">', unsafe_allow_html=True)
        
        # Date Range Selector
        timelapse_summary = get_timelapse_summary()
        
        if timelapse_summary['total_images'] > 0:
            # Get date range from data
            if timelapse_summary['date_range']:
                start_date = datetime.fromisoformat(timelapse_summary['date_range']['start']).date()
                end_date = datetime.fromisoformat(timelapse_summary['date_range']['end']).date()
            else:
                start_date = date(2015, 1, 1)
                end_date = date.today()
            
            st.markdown("**Dates ↑**")
            
            # Date range selector - use Streamlit's built-in range support
            date_range = st.date_input(
                "Select date range",
                value=(start_date, end_date),
                min_value=start_date,
                max_value=end_date,
                key="date_range_selector",
            )
            
            # Handle date range (can be tuple or single date)
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
            st.markdown("**Dates ↑**")
            st.info("No timelapse data")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Middle Row: Image Display
    st.markdown('<div class="image-section">', unsafe_allow_html=True)
    
    # Get images for selected date range
    if timelapse_summary['total_images'] > 0:
        engine = QueryEngine()
        
        # Query timeline for selected range
        timeline = engine.query('timeline', date_range=(selected_start.isoformat(), selected_end.isoformat()))
        
        if timeline.get('timeline'):
            # Show images from timeline
            images = find_timelapse_images()
            images_in_range = [
                (img_path, dt) for img_path, dt in images
                if selected_start <= dt.date() <= selected_end
            ]
            
            if images_in_range:
                # Show first and last images (before/after comparison)
                if len(images_in_range) >= 2:
                    img1_path, date1 = images_in_range[0]
                    img2_path, date2 = images_in_range[-1]
                    
                    col_img1, col_img2 = st.columns(2)
                    
                    with col_img1:
                        img1 = cv2.imread(str(img1_path))
                        if img1 is not None:
                            st.image(numpy_to_streamlit(img1), caption=f"{date1.strftime('%Y-%m')}", use_container_width=True)
                    
                    with col_img2:
                        img2 = cv2.imread(str(img2_path))
                        if img2 is not None:
                            st.image(numpy_to_streamlit(img2), caption=f"{date2.strftime('%Y-%m')}", use_container_width=True)
                    
                    st.markdown(f"**{date1.strftime('%Y-%m')} → {date2.strftime('%Y-%m')}**")
                else:
                    # Single image
                    img_path, dt = images_in_range[0]
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        st.image(numpy_to_streamlit(img), caption=f"{dt.strftime('%Y-%m-%d')}", use_container_width=True)
            else:
                st.info("No images in selected date range")
        else:
            st.info("Processing images... Run batch processor first.")
    else:
        # Fallback: show regular images
        available_images = list_available_images()
        if available_images:
            selected = st.selectbox("Select image", options=[img.name for img in available_images])
            img_path = available_images[[img.name for img in available_images].index(selected)]
            img = cv2.imread(str(img_path))
            if img is not None:
                st.image(numpy_to_streamlit(img), use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Bottom Row: Container Stats
    st.markdown('<div class="stats-section">', unsafe_allow_html=True)
    st.markdown("### container stats")
    
    if timelapse_summary['total_images'] > 0:
        # Get insights for date range
        engine = QueryEngine()
        
        # Inventory query
        inventory = engine.query('inventory', date_range=(selected_start.isoformat(), selected_end.isoformat()))
        
        # Changes query
        changes = engine.query('changes', date_range=(selected_start.isoformat(), selected_end.isoformat()))
        
        # Revenue query
        revenue = engine.query('revenue', date_range=(selected_start.isoformat(), selected_end.isoformat()))
        
        # Display stats
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        
        with col_s1:
            if inventory.get('snapshots'):
                avg_vol = sum(s.get('volume_m3', 0) for s in inventory['snapshots']) / len(inventory['snapshots'])
                st.metric("Avg Volume", f"{avg_vol:.1f} m³")
            else:
                st.metric("Avg Volume", "N/A")
        
        with col_s2:
            if changes.get('summary'):
                st.metric("Arrivals", changes['summary'].get('arrivals', 0))
            else:
                st.metric("Arrivals", "N/A")
        
        with col_s3:
            if changes.get('summary'):
                st.metric("Departures", changes['summary'].get('departures', 0))
            else:
                st.metric("Departures", "N/A")
        
        with col_s4:
            if revenue.get('total_value'):
                st.metric("Est. Value", f"${revenue['total_value']:,.0f}")
            else:
                st.metric("Est. Value", "N/A")
        
        # Timeline chart
        if timeline.get('values'):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=timeline['dates'],
                y=timeline['values'],
                mode='lines+markers',
                name='Volume',
                line=dict(color='#FF0000', width=2),
            ))
            fig.update_layout(
                title='Volume Over Time',
                xaxis_title='Date',
                yaxis_title='Volume (m³)',
                height=300,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select a date range to see container stats")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Close main container
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
