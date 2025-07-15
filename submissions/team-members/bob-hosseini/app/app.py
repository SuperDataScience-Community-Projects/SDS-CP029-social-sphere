import streamlit as st
import pickle
import pandas as pd
import numpy as np
import warnings
import logging
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from config_loader import get_config

# Import frontend helper
from frontend_helper import SocialSphereUI, PredictionUI

# Load configuration
config = get_config()

# Configure warnings and logging based on config
app_config = config.get_app_config()
if app_config.get('suppress_warnings', True):
    warnings.filterwarnings('ignore')

# Suppress MLflow warnings about version mismatches
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("sklearn").setLevel(logging.ERROR)

# Set page config from configuration
ui_config = config.get_ui_config()
st.set_page_config(
    page_title=ui_config.get('page_title', "SocialSphere Analytics"),
    layout=ui_config.get('layout', "wide"),
    initial_sidebar_state=ui_config.get('sidebar_state', "auto")
)

# Load data
@st.cache_data
def load_data():
    data_config = config.get_data_config()
    data_path = data_config.get('local_path', 'data/data_cleaned.pickle')
    with open(data_path, 'rb') as f:
        df = pickle.load(f)
    return df

def main():
    """Main application function"""
    # Load the data
    df = load_data()
    
    # Initialize UI components
    ui = SocialSphereUI(df)
    prediction_ui = PredictionUI(df)
    
    # Render sidebar
    ui.render_sidebar()
    
    # Main title
    st.title("📱 SocialSphere Analytics: Social Media Conflicts & Addiction Prediction")
    
    # Initialize session state for tab selection
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0
    
    # Create tabs with session state tracking
    tab_names = ["📊 Exploratory Data Analysis (EDA)", "🔮 Predicting Conflicts & Addiction"]
    
    # Use columns to create a custom tab selector that preserves state
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button(tab_names[0], type="secondary" if st.session_state.active_tab != 0 else "primary"):
            st.session_state.active_tab = 0
    
    with col2:
        if st.button(tab_names[1], type="secondary" if st.session_state.active_tab != 1 else "primary"):
            st.session_state.active_tab = 1
    
    # Render content based on active tab
    if st.session_state.active_tab == 0:
        st.markdown("---")
        ui.render_eda_tab()
    else:
        st.markdown("---")
        prediction_ui.render_prediction_tab()

if __name__ == "__main__":
    main()
