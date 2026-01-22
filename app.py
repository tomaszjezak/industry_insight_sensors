"""
Streamlit dashboard for C&D Debris Insights.
Interactive UI for analyzing construction debris pile images.
"""

import streamlit as st
import numpy as np
from pathlib import Path
import cv2
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Page config must be first Streamlit command
st.set_page_config(
    page_title="C&D Debris Insights",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import pipeline components
from src.scraper import list_available_images, DATA_DIR
from src.pipeline import DebrisAnalysisPipeline


# Custom CSS for industrial aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Space+Grotesk:wght@400;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #e94560 !important;
    }
    
    .metric-card {
        background: rgba(233, 69, 96, 0.1);
        border: 1px solid #e94560;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    }
    
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.5rem;
        font-weight: 700;
        color: #e94560;
    }
    
    .metric-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.9rem;
        color: #a0a0a0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stSidebar {
        background: linear-gradient(180deg, #16213e 0%, #1a1a2e 100%);
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #e94560 0%, #0f3460 100%);
        color: white;
        border: none;
        border-radius: 5px;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
    }
    
    .stButton>button:hover {
        background: linear-gradient(90deg, #0f3460 0%, #e94560 100%);
    }
    
    .info-box {
        background: rgba(15, 52, 96, 0.5);
        border-left: 4px solid #e94560;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(233, 69, 96, 0.1);
        border-radius: 8px 8px 0 0;
        color: #e94560;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .stTabs [aria-selected="true"] {
        background: #e94560 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


def get_pipeline(camera_height=None):
    """Initialize pipeline (not cached due to dynamic height)."""
    return DebrisAnalysisPipeline(
        use_sam=True,
        depth_model='small',
        camera_height_m=camera_height,  # None = auto-estimate
    )


def render_metric_card(label: str, value: str, unit: str = ""):
    """Render a styled metric card."""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}<span style="font-size: 1rem;">{unit}</span></div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def create_material_chart(materials: dict) -> go.Figure:
    """Create a styled pie chart for material breakdown."""
    colors = {
        'concrete': '#808080',
        'wood': '#8B4513',
        'metal': '#B87333',
        'mixed': '#4a4a4a',
        'plastic': '#4169E1',
        'vegetation': '#228B22',
    }
    
    labels = list(materials.keys())
    values = list(materials.values())
    chart_colors = [colors.get(m, '#666666') for m in labels]
    
    fig = go.Figure(data=[go.Pie(
        labels=[l.capitalize() for l in labels],
        values=values,
        hole=0.4,
        marker_colors=chart_colors,
        textinfo='label+percent',
        textfont=dict(family='Space Grotesk', size=14),
        hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
    )])
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Space Grotesk'),
        showlegend=False,
        margin=dict(t=20, b=20, l=20, r=20),
        height=300,
    )
    
    return fig


def numpy_to_streamlit(img: np.ndarray) -> np.ndarray:
    """Convert BGR numpy array to RGB for Streamlit display."""
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    # Header
    st.markdown("""
    # 🏗️ C&D Debris Insights
    ### AI-Powered Construction Debris Analysis
    """)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        auto_height = st.checkbox(
            "🤖 Auto-estimate camera height",
            value=True,
            help="Automatically estimate camera altitude from image (uses object detection)"
        )
        st.session_state['auto_height'] = auto_height
        
        if not auto_height:
            camera_height = st.slider(
                "Camera Height (m)",
                min_value=1.0,
                max_value=100.0,
                value=30.0,
                step=1.0,
                help="Height of camera - 3m for pole-mount, 30-100m for drone aerial"
            )
            st.session_state['camera_height'] = camera_height
        else:
            st.session_state['camera_height'] = None
            st.info("Height will be auto-detected from image")
        
        st.markdown("---")
        
        st.markdown("## 📁 Data Source")
        
        source = st.radio(
            "Select input method",
            ["Upload Image", "Sample Images"],
            index=1,
        )
        
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        
        available_images = list_available_images()
        st.metric("Available Images", len(available_images))
        
        if st.button("🔄 Refresh Image List"):
            st.cache_data.clear()
            st.rerun()
    
    # Main content
    col_input, col_output = st.columns([1, 1])
    
    with col_input:
        st.markdown("### 📷 Input Image")
        
        image = None
        image_name = None
        
        if source == "Upload Image":
            uploaded = st.file_uploader(
                "Drop image here",
                type=['jpg', 'jpeg', 'png', 'webp'],
                help="Upload an overhead image of a debris pile"
            )
            
            if uploaded:
                # Read uploaded file
                file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                image_name = uploaded.name
        
        else:  # Sample Images
            available_images = list_available_images()
            
            if available_images:
                # Image selector
                image_options = [img.name for img in available_images]
                selected = st.selectbox(
                    "Select sample image",
                    options=image_options,
                    index=0,
                )
                
                selected_path = available_images[image_options.index(selected)]
                image = cv2.imread(str(selected_path))
                image_name = selected
            else:
                st.warning("No sample images found. Run the scraper first:")
                st.code("python src/scraper.py", language="bash")
        
        # Display input image
        if image is not None:
            st.image(numpy_to_streamlit(image), caption=image_name, use_container_width=True)
            
            # Analyze button
            analyze_btn = st.button("🔍 Analyze Debris Pile", type="primary", use_container_width=True)
        else:
            st.info("👆 Select or upload an image to begin analysis")
            analyze_btn = False
    
    with col_output:
        st.markdown("### 📊 Analysis Results")
        
        if analyze_btn and image is not None:
            with st.spinner("Analyzing debris pile..."):
                try:
                    # Get camera height setting
                    if st.session_state.get('auto_height', True):
                        cam_height = None  # Auto-estimate
                    else:
                        cam_height = st.session_state.get('camera_height', 30.0)
                    
                    # Create pipeline
                    pipeline = get_pipeline(camera_height=cam_height)
                    
                    # Run analysis
                    result = pipeline.analyze(image, timestamp=datetime.now())
                    
                    # Store result in session state for comparison
                    st.session_state['last_result'] = result
                    st.session_state['result_ready'] = True
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.session_state['result_ready'] = False
        
        # Display results if available
        if st.session_state.get('result_ready', False):
            result = st.session_state['last_result']
            
            # Show camera height info
            if result.get('height_auto_estimated'):
                height_info = result.get('height_estimation', {})
                st.markdown(f"""
                <div class="info-box">
                🤖 <b>Auto-detected camera height:</b> {result['camera_height_m']:.1f}m 
                ({height_info.get('category', 'unknown').replace('_', ' ')}, 
                confidence: {height_info.get('confidence', 0)*100:.0f}%)
                </div>
                """, unsafe_allow_html=True)
            
            # Metrics row
            m1, m2, m3, m4 = st.columns(4)
            
            with m1:
                render_metric_card("Volume", f"{result['volume']['volume_m3']:.1f}", " m³")
            
            with m2:
                render_metric_card("Area", f"{result['volume']['pile_area_m2']:.1f}", " m²")
            
            with m3:
                render_metric_card("Avg Height", f"{result['volume']['avg_height_m']:.2f}", " m")
            
            with m4:
                render_metric_card("Est. Tonnage", f"{result['volume']['tonnage_estimate']:.0f}", " t")
            
            # Tabs for detailed views
            tab1, tab2, tab3, tab4 = st.tabs([
                "🎨 Segmentation",
                "📏 Depth Map", 
                "🧱 Materials",
                "📋 Full Report"
            ])
            
            with tab1:
                vis = result.get('_visualizations', {})
                if 'segmentation' in vis and vis['segmentation'] is not None:
                    st.image(
                        numpy_to_streamlit(vis['segmentation']),
                        caption=f"Pile Segmentation ({result['segmentation']['method']})",
                        use_container_width=True
                    )
                    st.info(f"Detected {result['segmentation']['num_segments']} segment(s)")
            
            with tab2:
                vis = result.get('_visualizations', {})
                col_d1, col_d2 = st.columns(2)
                
                with col_d1:
                    if 'depth' in vis and vis['depth'] is not None:
                        st.image(
                            numpy_to_streamlit(vis['depth']),
                            caption="Depth Map (Turbo colormap)",
                            use_container_width=True
                        )
                
                with col_d2:
                    if 'height' in vis and vis['height'] is not None:
                        st.image(
                            numpy_to_streamlit(vis['height']),
                            caption="Height Map (pile region)",
                            use_container_width=True
                        )
                
                st.markdown(f"""
                <div class="info-box">
                <b>Height Statistics:</b><br>
                • Max Height: {result['volume']['max_height_m']:.2f} m<br>
                • Avg Height: {result['volume']['avg_height_m']:.2f} m<br>
                • Camera Height: {result['camera_height_m']:.1f} m
                </div>
                """, unsafe_allow_html=True)
            
            with tab3:
                vis = result.get('_visualizations', {})
                
                # Show spatial material map (where each material is)
                if 'materials_spatial' in vis and vis['materials_spatial'] is not None:
                    st.image(
                        numpy_to_streamlit(vis['materials_spatial']),
                        caption="Spatial Material Map (only within detected piles/containers)",
                        use_container_width=True
                    )
                
                col_m1, col_m2 = st.columns([1, 1])
                
                with col_m1:
                    if 'materials' in vis and vis['materials'] is not None:
                        st.image(
                            numpy_to_streamlit(vis['materials']),
                            caption="Material Overlay",
                            use_container_width=True
                        )
                
                with col_m2:
                    # Pie chart
                    fig = create_material_chart(result['materials']['percentages'])
                    st.plotly_chart(fig, use_container_width=True)
                
                st.success(f"**Dominant Material:** {result['materials']['dominant'].capitalize()} ({result['materials']['dominant_percentage']:.1f}%)")
                
                st.markdown("""
                <div class="info-box">
                <b>Note:</b> Materials are only classified within the detected pile/container regions (green boundary).
                Roads, ground, and background are excluded from analysis.
                </div>
                """, unsafe_allow_html=True)
            
            with tab4:
                # JSON output
                st.markdown("#### 📄 Analysis JSON")
                
                pipeline = get_pipeline()
                json_output = pipeline.to_json(result)
                
                st.code(json_output, language='json')
                
                st.download_button(
                    "📥 Download JSON",
                    data=json_output,
                    file_name=f"debris_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                )
        else:
            st.info("👈 Click **Analyze Debris Pile** to start")
    
    # Change Detection Section
    st.markdown("---")
    st.markdown("### 🔄 Change Detection")
    
    with st.expander("Compare Two Images", expanded=False):
        col_c1, col_c2 = st.columns(2)
        
        available_images = list_available_images()
        
        if len(available_images) >= 2:
            image_options = [img.name for img in available_images]
            
            with col_c1:
                st.markdown("**Image 1 (Before)**")
                img1_name = st.selectbox(
                    "Select first image",
                    options=image_options,
                    index=0,
                    key="change_img1"
                )
                img1_path = available_images[image_options.index(img1_name)]
                img1 = cv2.imread(str(img1_path))
                st.image(numpy_to_streamlit(img1), use_container_width=True)
            
            with col_c2:
                st.markdown("**Image 2 (After)**")
                img2_name = st.selectbox(
                    "Select second image",
                    options=image_options,
                    index=min(1, len(image_options)-1),
                    key="change_img2"
                )
                img2_path = available_images[image_options.index(img2_name)]
                img2 = cv2.imread(str(img2_path))
                st.image(numpy_to_streamlit(img2), use_container_width=True)
            
            if st.button("🔍 Detect Changes", use_container_width=True):
                with st.spinner("Comparing images..."):
                    # Get camera height setting
                    if st.session_state.get('auto_height', True):
                        cam_height = None
                    else:
                        cam_height = st.session_state.get('camera_height', 30.0)
                    
                    pipeline = get_pipeline(camera_height=cam_height)
                    
                    # Analyze both images
                    result1 = pipeline.analyze(img1)
                    result2 = pipeline.analyze(img2, previous_result=result1)
                    
                    if result2['changes']:
                        changes = result2['changes']
                        
                        # Display change metrics
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.metric("Change Type", changes['change_type'].replace('_', ' ').title())
                        with c2:
                            st.metric("Area Changed", f"{changes['change_percentage']}%")
                        with c3:
                            delta = changes['pile_net_change_pct']
                            st.metric(
                                "Pile Size", 
                                f"{abs(delta):.1f}%",
                                delta=f"{'↑' if delta > 0 else '↓'} {'grew' if delta > 0 else 'shrank'}"
                            )
                        
                        # Show change visualization
                        if '_change_vis' in result2:
                            st.image(
                                numpy_to_streamlit(result2['_change_vis']),
                                caption="Change Visualization (red = changed areas)",
                                use_container_width=True
                            )
        else:
            st.warning("Need at least 2 images for change detection. Add more images to the data folder.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        C&D Debris Insights • Phase 1 MVP • Built for San Diego C&D Recycling
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    # Initialize session state
    if 'result_ready' not in st.session_state:
        st.session_state['result_ready'] = False
    if 'camera_height' not in st.session_state:
        st.session_state['camera_height'] = None
    if 'auto_height' not in st.session_state:
        st.session_state['auto_height'] = True
    
    main()
