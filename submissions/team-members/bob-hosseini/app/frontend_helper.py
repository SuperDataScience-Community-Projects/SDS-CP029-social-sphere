"""
Frontend Helper for SocialSphere Analytics App

This module contains all the UI components and helper functions for the Streamlit app,
making the main app.py file cleaner and more maintainable.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import shap
import mlflow
import warnings

import sys
import os
# Add the src directory to the path (relative to the app directory)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import regression
import utils  # Add this import
from config_loader import get_config

warnings.filterwarnings('ignore')

# Load configuration
config = get_config()


class SocialSphereUI:
    """Main UI class for SocialSphere Analytics app"""
    
    def __init__(self, df):
        """Initialize with the main dataframe"""
        self.df = df
    
    def render_sidebar(self):
        """Render the sidebar with project information"""
        with st.sidebar:
            st.markdown("### ℹ️ Project Overview")
            st.markdown(
                """
                **SocialSphere Analytics** is a comprehensive analysis platform for understanding social media usage patterns, 
                conflicts, and addiction scores among students. This app provides both exploratory data analysis and 
                predictive modeling capabilities.
                    
                > 📊 **Features:**
                > - Interactive EDA with multiple visualizations
                > - Conflict prediction based on user characteristics
                > - Addiction score forecasting
                > - Real-time data insights
                """
            )

            st.markdown("### 📊 Dataset Summary")
            st.markdown(
                f"""
                - **Total Records:** {len(self.df):,}
                - **Features:** {len(self.df.columns)}
                - **Countries:** {self.df['Country'].nunique()}
                - **Platforms:** {self.df['Platform'].nunique()}
                """
            )

            st.markdown("### 🎯 Target Variables")
            st.markdown(
                """
                - **Conflicts:** Number of conflicts (0-5)
                - **Addicted_Score:** Addiction level (1-10)
                """
            )

            st.markdown("### 🤖 MLflow Models")
            
            # Get model configurations
            conflicts_config = config.get_model_config('conflicts')
            addiction_config = config.get_model_config('addiction')
            
            st.markdown(
                f"""
                **Pre-trained Models:**
                - **Conflicts:** {conflicts_config.get('name', 'CatBoost Binary Classifier')}
                - **Addiction:** {addiction_config.get('name', 'CatBoost Regressor with Rounding')}
                
                Models include full preprocessing pipelines - no manual encoding required!
                """
            )
            
            # Add MLflow dashboard link from config
            mlflow_config = config.get_mlflow_config()
            experiment_url = mlflow_config.get('experiment_url')
            if experiment_url:
                st.markdown(f"[View MLflow Experiments and Models]({experiment_url})")

            st.markdown("### 📁 Dataset Source")
            
            # Add data source link from config
            data_config = config.get_data_config()
            source_url = data_config.get('source_url')
            if source_url:
                st.markdown(f"[Students' Social Media Addiction Dataset]({source_url})")

            st.markdown("### 🗂️ GitHub Repository")
            st.markdown(
                "[View on GitHub](https://github.com/SuperDataScience-Community-Projects/SDS-CP029-social-sphere/tree/main/submissions/team-members/bob-hosseini)"
            )

    
    def render_eda_tab(self):
        """Render the Exploratory Data Analysis tab"""
        st.header("📊 Exploratory Data Analysis")
        
        # Data overview
        self._render_data_overview()
        
        # Descriptive statistics
        self._render_descriptive_stats()
        
        # Numeric features distribution
        self._render_numeric_distributions()
        
        # Geographic and platform analysis
        self._render_geographic_platform_analysis()
        
        # Target variables analysis
        self._render_target_variables()
        
        # Gender and academic level analysis
        self._render_demographic_analysis()
        
        # Correlation analysis
        self._render_correlation_analysis()
        
        # Relationship status analysis
        self._render_relationship_analysis()
    
    def _render_data_overview(self):
        """Render dataset overview section"""
        st.subheader("📋 Dataset Overview")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(self.df.head(100))
        
        with col2:
            st.markdown("**Dataset Info:**")
            st.write(f"Shape: {self.df.shape}")
            st.write(f"Missing values: {self.df.isnull().sum().sum()}")
            
            st.markdown("**Numeric Columns:**")
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            st.write(", ".join(numeric_cols))
            
            st.markdown("**Categorical Columns:**")
            categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
            st.write(", ".join(categorical_cols))
    
    def _render_descriptive_stats(self):
        """Render descriptive statistics section"""
        st.subheader("📈 Descriptive Statistics")
        st.dataframe(self.df.describe())
    
    def _render_numeric_distributions(self):
        """Render numeric features distribution"""
        st.subheader("📊 Distribution of Numeric Features")
        numeric_features = ['Age', 'Daily_Usage', 'Sleep_Hrs', 'Mental_Health']

        # Prepare data in long format for a single box plot
        df_long = self.df[numeric_features].melt(var_name='Feature', value_name='Value')

        fig = px.box(
            df_long,
            y='Feature',
            x='Value',
            orientation='h',
            color='Feature',
            title="Distribution of Numeric Features",
            color_discrete_sequence=px.colors.qualitative.Dark24_r
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_geographic_platform_analysis(self):
        """Render geographic and platform analysis"""
        # Count plots of countries and platforms
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌍 Top Countries")
            country_counts = self.df['Country'].value_counts().head(11)
            fig = px.bar(x=country_counts.values, y=country_counts.index, 
                         orientation='h', title=f"Top 11 Countries out of {len(self.df['Country'].unique())}")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📱 Social Media Platforms")
            platform_counts = self.df['Platform'].value_counts()
            fig = px.pie(values=platform_counts.values, names=platform_counts.index, 
                         title="Most Used Platforms")
            st.plotly_chart(fig, use_container_width=True)

        # Add insights about countries and platforms
        st.markdown("""
        **🌍 Geographic Distribution:**
        - **Majority** of participants are from **India, USA, and Canada**
        - **~80 countries** have only **1 participant** each
        - **Potential bias**: Geographic over-representation may affect model generalization
        - **Recommendation**: Consider grouping countries into regions or using frequency encoding

        **📱 Platform Usage:**
        - **Instagram, Facebook, TikTok** are the most popular platforms
        - **YouTube** is the least used platform (only ~10 users)
        - **Long-tail distribution** may cause overfitting in modeling
        - **Recommendation**: Group platforms by type (social media, messaging, etc.) or trim long tail
        """)
    
    def _render_target_variables(self):
        """Render target variables analysis"""
        st.subheader("🎯 Distributions of Target Variables")

        col1, col2 = st.columns(2)

        with col1:
            fig_conflicts = px.histogram(
                self.df, 
                x='Conflicts', 
                nbins=6,
                title="Conflicts Over Social Media",
                color_discrete_sequence=['blue'],
                marginal='violin'
            )
            fig_conflicts.update_layout(
                bargap=0.3,
                width=350,
                height=350
            )
            st.plotly_chart(fig_conflicts, use_container_width=True)
            st.markdown("""
            **Conflict Distribution Analysis:**
            - Few students report very low (0,1) or very high (5) conflicts over social media.
            - 3 conflicts is the most common level of conflicts over social media.        
            """)

        with col2:
            fig_addicted = px.histogram(
                self.df,
                x='Addicted_Score',
                nbins=10,
                title="Addicted Score",
                color_discrete_sequence=['red'],
                marginal='violin'
            )
            fig_addicted.update_layout(
                bargap=0.3,
                width=350,
                height=350
            )
            st.plotly_chart(fig_addicted, use_container_width=True)
            st.markdown("""
            **Addicted Score Distribution Analysis:**
            - Most students have moderate Addicted Scores.
            - Very high (9,10) and very low (1,2) Addicted Scores are less common.
            - 7 is the most common Addicted Score while 6 is among least common reported Scores.        
            """)
    
    def _render_demographic_analysis(self):
        """Render demographic analysis"""
        # Distribution of conflicts by gender
        st.subheader("📊 Distribution of Conflicts by Gender")
        fig = px.box(self.df, x='Gender', y='Conflicts', 
                     title="Conflicts Over Social Media by Gender",
                     color='Gender',
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Insights:**
        - The median conflicts are the same for males and females.
        - Female participants have more variability in their conflicts.
        """)

        # Distribution of addiction score by academic level
        st.subheader("📊 Distribution of Addiction Score by Academic Level")
        fig = px.box(self.df, x='Academic_Level', y='Addicted_Score', 
                     title="Addiction Score by Academic Level",
                     color='Academic_Level',
                     color_discrete_sequence=px.colors.qualitative.Alphabet)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Insights:**
        - The median addicted score is similar across academic levels, generally between 7 and 8.
        - High school students show more outliers and tend to have higher addicted scores overall.
        - This may be influenced by the under-representation of high school students in the dataset.
        """)

        # Box plots of countries and platforms
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌍 Conflicts by Top Countries")
            top_countries = self.df['Country'].value_counts().head(11).index
            df_top_countries = self.df[self.df['Country'].isin(top_countries)]
            fig = px.box(df_top_countries, x='Country', y='Conflicts',
                         title="Conflicts by Top Countries",
                         color='Country',
                         color_discrete_sequence=px.colors.qualitative.Dark24)
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📱 Addicted Score by Platform")
            fig = px.box(self.df, x='Platform', y='Addicted_Score',
                         title="Addicted Score by Social Media Platform",
                         color='Platform',
                         color_discrete_sequence=px.colors.qualitative.Dark2)
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

        # Add insights about platform and country patterns
        st.markdown("""
    **🌍 Geographic Conflict Trends:**
    - USA shows the **highest conflicts** scores with a median of 4 (maximum).   
    - India, Turkey, Mexico, Spain, and UK show similar spread of conflicts scores. They are in the **2nd tier** after USA.            
    - Ireland, Denmark, Switzerland, and Canada show a **tight spread** in conflicts over social media.
                    
    **📱 Platform Addiction Patterns:**
    - **TikTok** has the highest median addiction score (8) with wide spread and outliers
    - **Instagram & WhatsApp** show high median scores (7) with broad ranges
    - **Instagram & Twitter** users show the widest range of addiction scores
    - **Less popular platforms** (Snapchat, WeChat, Line) have insufficient data for strong conclusions
    """)
    
    def _render_correlation_analysis(self):
        """Render correlation analysis"""
        st.subheader("🔗 Correlation Matrix of Numeric Features")
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            numeric_df.corr(),
            annot=True,
            cmap='coolwarm',
            center=0,
            ax=ax,
            annot_kws={"size": 8}
        )
        plt.title("Correlation Matrix of Numeric Features", fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
        st.pyplot(fig, use_container_width=True)
        
        st.markdown("""
        **Key Correlations:**
        - **Conflicts** and **Addicted_Score** both show strong correlations with several key features:
            - **Daily_Usage**: Positive correlation (higher usage is linked to more conflicts and higher addiction scores)
            - **Mental_Health**: Negative correlation (lower mental health scores are associated with higher conflicts and addiction)
            - **Sleep_Hrs**: Negative correlation (less sleep is linked to more conflicts and higher addiction)
        """)
    
    def _render_relationship_analysis(self):
        """Render relationship status analysis"""
        st.subheader("💕 Conflicts by Relationship Status")
        fig = px.box(self.df, x='Relationship_Status', y='Conflicts',
                     title="Conflicts Over Social Media by Relationship Status",
                     color='Relationship_Status',
                     color_discrete_sequence=px.colors.qualitative.Alphabet)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Insights:**
        - All three relationship status groups have a similar median number of conflicts (around 3).
        - In Relationship and Complicated groups have a slightly narrower range, showing fewer students experience no conflicts in these groups.
        """)


class ModelManager:
    """Manages MLflow model loading and caching"""

    @staticmethod
    @st.cache_resource
    def load_conflicts_model_sklearn():
        """Load the conflicts classification model as sklearn pipeline for SHAP"""
        try:
            conflicts_config = config.get_model_config('conflicts')
            model_uri = conflicts_config.get('pyfunc_uri')
            if not model_uri:
                raise ValueError("Conflicts sklearn model URI not found in configuration")
            pipe = mlflow.sklearn.load_model(model_uri)
            return pipe
        except Exception as e:
            st.error(f"Error loading conflicts sklearn model: {e}")
            return None

    @staticmethod
    @st.cache_resource
    def load_addiction_model_sklearn():
        """Load the addiction score regression model as sklearn pipeline for SHAP"""
        try:
            addiction_config = config.get_model_config('addiction')
            model_uri = addiction_config.get('pyfunc_uri')
            if not model_uri:
                raise ValueError("Addiction sklearn model URI not found in configuration")
            pipe = mlflow.sklearn.load_model(model_uri)
            return pipe
        except Exception as e:
            st.error(f"Error loading addiction sklearn model: {e}")
            return None


class PredictionUI:
    """Handles prediction-related UI components"""
    
    def __init__(self, df):
        self.df = df
        self.model_manager = ModelManager()
    
    def render_prediction_tab(self):
        """Render the prediction tab"""
        st.header("🔮 Prediction Models")
        
        # Load models
        # conflicts_model = self.model_manager.load_conflicts_model()
        # addiction_model = self.model_manager.load_addiction_model()
        conflicts_model_sklearn = self.model_manager.load_conflicts_model_sklearn()
        addiction_model_sklearn = self.model_manager.load_addiction_model_sklearn()
        
        # Display model loading status
        self._render_model_status(conflicts_model_sklearn, 
                                addiction_model_sklearn)
        
        if conflicts_model_sklearn is None and addiction_model_sklearn is None:
            self._render_model_error()
            return
        
        # Single combined prediction interface
        self._render_combined_prediction(conflicts_model_sklearn,
                                       addiction_model_sklearn)
        
        # Model information
        self._render_model_info()
    
    def _render_model_status(self, conflicts_model_sklearn, addiction_model_sklearn):
        """Render model loading status"""
        st.markdown("### 🔧 Model Status")
        col1, col2 = st.columns(2)

        with col1:
            if conflicts_model_sklearn is not None:
                st.success("✅ Conflicts Sklearn: Loaded")
            else:
                st.error("❌ Conflicts Sklearn: Failed")

        with col2:
            if addiction_model_sklearn is not None:
                st.success("✅ Addiction Sklearn: Loaded")
            else:
                st.error("❌ Addiction Sklearn: Failed")

    def _render_model_error(self):
        """Render model loading error"""
        st.error("""
        **⚠️ No models could be loaded!**
        
        Please ensure:
        1. MLflow tracking URI is correctly set
        2. Models are registered in the MLflow model registry
        3. Required dependencies are installed
        
        **Expected models:**
        - `conflict_catboost_multiclass`
        - `addicted_score_catboost_all_features+rounded`
        """)
        st.stop()
    
    def _render_combined_prediction(self, conflicts_model_sklearn, addiction_model_sklearn):
        """Render combined prediction interface for both conflicts and addiction"""
        st.subheader("🔮 Predict Social Media Impact")
        st.markdown("Get predictions for both **conflicts** and **addiction score** based on your profile.")
        
        # User input form
        st.markdown("### 📝 Enter User Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", min_value=16, max_value=30, value=20)
            gender = st.selectbox("Gender", options=self.df['Gender'].unique())
            academic_level = st.selectbox("Academic Level", options=self.df['Academic_Level'].unique())
            country = st.selectbox(
                "Country",
                options=sorted(self.df['Country'].unique()),
                index=sorted(self.df['Country'].unique()).index('USA') if 'USA' in self.df['Country'].unique() else 0
            )
            avg_daily_usage = st.slider("Daily Usage (Hours)", min_value=1.0, max_value=10.0, value=5.0, step=0.1)
        
        with col2:
            platform = st.selectbox("Most Used Platform", options=self.df['Platform'].unique())
            affects_academic = st.selectbox("Affects Academic Performance", options=self.df['Academic_Affects'].unique())
            sleep_hours = st.slider("Sleep Hours Per Night", min_value=3.0, max_value=10.0, value=7.0, step=0.1)
            relationship_status = st.selectbox("Relationship Status", options=self.df['Relationship_Status'].unique())
        
        # Automatically make predictions (no button needed)
        # First predict conflicts (needed for addiction prediction)
        mental_health = 0 # just a dummy value for now
        conflicts_input_df = self._create_conflicts_input_df(
            age, gender, academic_level, country, avg_daily_usage, 
            platform, affects_academic, sleep_hours, mental_health, relationship_status
        )
        
        # Get conflicts prediction
        if conflicts_model_sklearn is not None:
            conflicts_pred = conflicts_model_sklearn.predict(conflicts_input_df)[0]
            if isinstance(conflicts_pred, np.ndarray):
                conflicts_pred = conflicts_pred.item()
        else:
            st.warning("⚠️ Conflicts model not available, using default value (3) for addiction prediction")
            conflicts_pred = 3
        
        # Create addiction input with predicted conflicts
        addiction_input_df = self._create_addiction_input_df(
            age, gender, academic_level, country, avg_daily_usage, 
            platform, affects_academic, sleep_hours, mental_health, 
            relationship_status, conflicts_pred
        )
        
        # Display results
        st.markdown("## 🎯 Prediction Results")
        
        # Results in columns
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            st.markdown("### 💥 Conflicts Prediction")
            if conflicts_model_sklearn is not None:
                self._display_conflicts_result(conflicts_pred)
            else:
                st.error("❌ Conflicts model not available")
        
        with result_col2:
            st.markdown("### 🎮 Addiction Score Prediction")
            if addiction_model_sklearn is not None:
                addiction_pred = addiction_model_sklearn.predict(addiction_input_df)[0]
                self._display_addiction_result(addiction_pred)
            else:
                st.error("❌ Addiction model not available")
        
        # SHAP explanations
        st.markdown("## 🔍 AI Model Explanations")
        st.markdown("""
        Understanding which factors influenced your predictions:
        - **Bar length** shows the magnitude of feature impact on the prediction
        """)
        
        shap_col1, shap_col2 = st.columns(2)
        
        with shap_col1:
            st.markdown("### 💥 Conflicts Model Explanation")
            if conflicts_model_sklearn is not None:
                self._render_shap_explanation(conflicts_model_sklearn, conflicts_input_df, "classification")
            else:
                st.warning("⚠️ Conflicts SHAP explanation not available")
        
        with shap_col2:
            st.markdown("### 🎮 Addiction Model Explanation")
            if addiction_model_sklearn is not None:
                self._render_shap_explanation(addiction_model_sklearn, addiction_input_df, "regression")
            else:
                st.warning("⚠️ Addiction SHAP explanation not available")
    
    def _create_conflicts_input_df(self, age, gender, academic_level, country, 
                                 avg_daily_usage, platform, affects_academic, 
                                 sleep_hours, mental_health, relationship_status):
        """Create input DataFrame for conflicts prediction"""
        return pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'Academic_Level': academic_level,
            'Country': country,
            'Daily_Usage': avg_daily_usage,
            'Platform': platform,
            'Academic_Affects': affects_academic,
            'Sleep_Hrs': sleep_hours,
            'Mental_Health': mental_health,
            'Relationship_Status': relationship_status
        }])
    
    def _create_addiction_input_df(self, age, gender, academic_level, country, 
                                 avg_daily_usage, platform, affects_academic, 
                                 sleep_hours, mental_health, relationship_status, conflicts):
        """Create input DataFrame for addiction prediction"""
        return pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'Academic_Level': academic_level,
            'Country': country,
            'Daily_Usage': avg_daily_usage,
            'Platform': platform,
            'Academic_Affects': affects_academic,
            'Sleep_Hrs': sleep_hours,
            'Mental_Health': mental_health,
            'Relationship_Status': relationship_status,
            'Conflicts': conflicts
        }])
    
    def _display_conflicts_result(self, conflict_value):
        """Display conflicts prediction result"""
        if isinstance(conflict_value, np.ndarray):
            conflict_value = conflict_value.item()
        conflict_class = "Low" if conflict_value == 0 else "High"
        st.success(f"🎯 **Predicted Conflict Class:** {conflict_class}")
        
    
    def _display_addiction_result(self, addiction_value):
        """Display addiction prediction result"""
        if isinstance(addiction_value, np.ndarray):
            addiction_value = addiction_value.item()
        
        # Addiction level interpretation
        addiction_level = "Low" if addiction_value < 3 else "Moderate" if addiction_value < 6 else "High"
        st.success(f"🎯 **Predicted Addiction Score:** {addiction_value:.1f}/10 ({addiction_level})")
        # st.markdown(f"**Addiction Level:** :{color}[{addiction_level}]")
    
    def _render_shap_explanation(self, model_sklearn, sample, model_type, shap_type=None, plot_type=None):
        """Render SHAP explanation using run_shap_experiment from utils.py"""
        if model_sklearn is not None:
            try:
                # Get SHAP configuration
                shap_config = config.get_shap_config()
                # app_config = config.get_app_config()
                
                # Use provided parameters or defaults from config
                if shap_type is None:
                    shap_type = shap_config.get('default_shap_type', 'tree')
                if plot_type is None:
                    plot_type = shap_config.get('default_plot_type', 'bar')
                
                # Get figure size from config
                fig_width = shap_config.get('figure_size', {}).get('width', 10)
                fig_height = shap_config.get('figure_size', {}).get('height', 6)
                
                # Create SHAP explanation using the utils function
                fig_shap = utils.run_shap_experiment(
                    best_model=model_sklearn,
                    X_train_full=sample,
                    # random_state=app_config.get('random_state', 42),
                    plot_type=plot_type,
                    shap_type=shap_type,
                    model_type=model_type,
                    figsize=(fig_width, fig_height)
                )
                
                st.pyplot(fig_shap, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error generating SHAP plot: {e}")
                st.error("Please ensure the model pipeline contains the expected named steps.")
        else:
            st.warning("⚠️ SHAP visualization requires sklearn models. Please ensure models are loaded correctly.")
    
    def _render_model_info(self):
        """Render model information section"""
        st.subheader("🎯 Model Information")
        st.markdown("""
        **MLflow Models Used:**
        - **Conflicts Prediction:** `conflict_catboost_multiclass` (CatBoost Classifier)
        - **Addiction Score Prediction:** `addicted_score_catboost_all_features+rounded` (CatBoost Regressor)
                
        - Models are loaded directly from the MLflow model registry
        - They include all necessary preprocessing steps.
        """) 