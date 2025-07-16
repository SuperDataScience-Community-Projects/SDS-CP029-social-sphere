## SocialSphere Analytics - Modal Deployment

def main():
    import streamlit as st
    import pickle
    # import pandas as pd
    # import numpy as np
    import warnings
    import logging
    import sys
    import os
    # from PIL import Image

    loading_placeholder = st.empty()
    loading_placeholder.info("🚀 Loading SocialSphere Analytics System... Please wait.")
    
    # Add the src directory to the path
    sys.path.append('/root/src')  # Modal deployment path
    sys.path.append('/root/app')  # For frontend_helper
    
    from config_loader import get_config
    from frontend_helper import SocialSphereUI, PredictionUI
    
    loading_placeholder.empty()

    # ====================================
    # Main App Content (only shown after loading)
    # ====================================

    # Adjust sidebar width
    st.markdown(
        """
        <style>
            section[data-testid="stSidebar"] {
                width: 400px !important; /* Adjust this value as needed */
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ====================================
    # Initialize and clear problematic state on startup
    # ====================================
    if 'app_initialized' not in st.session_state:
        # Clear any problematic session state
        for key in list(st.session_state.keys()):
            if any(word in key.lower() for word in ['file', 'upload', 'media', 'image']):
                del st.session_state[key]
        
        # Clear all cached data to prevent issues
        if hasattr(st, 'cache_data'):
            st.cache_data.clear()
        if hasattr(st, 'cache_resource'):
            st.cache_resource.clear()
        
        st.session_state.app_initialized = True
        st.rerun()  # Single page reload for all cleanup    

    # ====================================
    # Configuration and logging setup
    # ====================================
    
    # Load configuration using absolute path for Modal
    config = get_config()
    
    # Configure warnings and logging based on config
    app_config = config.get_app_config()
    if app_config.get('suppress_warnings', True):
        warnings.filterwarnings('ignore')

    # Suppress MLflow warnings about version mismatches
    logging.getLogger("mlflow").setLevel(logging.ERROR)
    logging.getLogger("sklearn").setLevel(logging.ERROR)
    
    # Setup logger for this module
    logger = logging.getLogger(__name__)

    # ====================================
    # Initialize Streamlit session state variables
    # ====================================
    def initialize_session_state() -> None:
        """
        Initialize necessary session state variables for Streamlit.
        """
        st.session_state.setdefault("debug", os.getenv("DEBUG_MODE", "false").lower() == "true")
        st.session_state.setdefault("df", None)
        st.session_state.setdefault("ui", None)
        st.session_state.setdefault("prediction_ui", None)
        st.session_state.setdefault("active_tab", 0)
        st.session_state.setdefault("data_loaded", False)
        logger.debug("Session state initialized.")
    
    # Initialize session state variables
    initialize_session_state()
    logger.debug("Session state initialized.")

    # ====================================
    # Set page configuration
    # ====================================
    ui_config = config.get_ui_config()
    st.set_page_config(
        page_title=ui_config.get('page_title', "SocialSphere Analytics"),
        layout=ui_config.get('layout', "wide"),
        initial_sidebar_state=ui_config.get('sidebar_state', "auto")
    )

    # ====================================
    # Data loading with caching
    # ====================================
    @st.cache_data
    def load_data():
        """Load and cache the main dataset"""
        try:
            data_config = config.get_data_config()
            # Use absolute path for Modal deployment
            data_path = '/root/data/data_cleaned.pickle'
            
            if not os.path.exists(data_path):
                # Fallback to relative path
                data_path = data_config.get('local_path', 'data/data_cleaned.pickle')
            
            with open(data_path, 'rb') as f:
                df = pickle.load(f)
            
            logger.info(f"Data loaded successfully from {data_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            st.error(f"Error loading data: {e}")
            return None

    # ====================================
    # Display header image
    # ====================================
    image_path = "/root/image.jpg"
    try:
        if os.path.exists(image_path):
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image_path, width=800)
        else:
            # Fallback to relative path
            fallback_path = os.path.join(os.path.dirname(__file__), '..', 'image.jpg')
            if os.path.exists(fallback_path):
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image(fallback_path, width=800)
    except Exception as e:
        logger.error(f"Error displaying image: {e}")
        st.write(f"Error displaying image: {e}")

    # ====================================
    # Main UI Layout
    # ====================================
    st.title("📱 SocialSphere Analytics:")
    st.title("Social Media Conflicts & Addiction Prediction")
    

    # ====================================
    # Check debug mode
    # ====================================
    if st.session_state.debug:
        st.warning("DEBUG MODE is ON")
        logger.debug("Debug mode is enabled.")

    # ====================================
    # Load data and initialize UI components
    # ====================================
    if not st.session_state.data_loaded:
        df = load_data()
        if df is not None:
            st.session_state.df = df
            st.session_state.ui = SocialSphereUI(df)
            st.session_state.prediction_ui = PredictionUI(df)
            st.session_state.data_loaded = True
            st.success("Data loaded and UI components initialized successfully!")
        else:
            st.error("Failed to load data. Please check the data files.")
            st.stop()

    # Get UI components from session state
    ui = st.session_state.ui
    prediction_ui = st.session_state.prediction_ui
    
    # Render sidebar
    ui.render_sidebar()

    # ====================================
    # Tab navigation with session state
    # ====================================
    tab_names = ["📊 Exploratory Data Analysis (EDA)", "🔮 Predicting Conflicts & Addiction"]
    
    # Use columns to create a custom tab selector that preserves state
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button(tab_names[0], type="secondary" if st.session_state.active_tab != 0 else "primary"):
            st.session_state.active_tab = 0
    
    with col2:
        if st.button(tab_names[1], type="secondary" if st.session_state.active_tab != 1 else "primary"):
            st.session_state.active_tab = 1

    # ====================================
    # Render content based on active tab
    # ====================================
    if st.session_state.active_tab == 0:
        st.markdown("---")
        # st.header("📊 Exploratory Data Analysis")
        # st.write("Explore comprehensive visualizations and insights from the social media dataset.")
        ui.render_eda_tab()
        logger.debug("Rendered EDA tab.")
    else:
        st.markdown("---")
        # st.header("🔮 Predictions")
        # st.write("Use machine learning models to predict social media conflicts and addiction scores.")
        prediction_ui.render_prediction_tab()
        logger.debug("Rendered prediction tab.")

    # ====================================
    # Footer information
    # ====================================
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>            
            <p>Advanced Social Media Behavior Analysis & Prediction</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()