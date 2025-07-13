import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import warnings
import os
import sys
sys.path.append('src')
import regression
import utils

warnings.filterwarnings('ignore')

# from dotenv import load_dotenv
# import dagshub
# load_dotenv()

# DAGSHUB_USER_NAME = os.getenv("DAGSHUB_USER_NAME")
# DAGSHUB_REPO = os.getenv("DAGSHUB_REPO")
# DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")
# os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

# import dagshub
# try:    
#     dagshub.init(repo_owner=DAGSHUB_USER_NAME, repo_name=DAGSHUB_REPO, mlflow=True)
#     print("Dagshub tracking enabled")
# except Exception as e:
#     print(f"Error initializing Dagshub: {e}")

# Suppress MLflow warnings about version mismatches
import logging
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("sklearn").setLevel(logging.ERROR)

# Set MLflow tracking URI (adjust as needed for your setup)
# mlflow.set_tracking_uri("file:../mlruns")

# Set page config
st.set_page_config(
    page_title="SocialSphere Analytics",
    layout="wide",
    initial_sidebar_state="auto"
)

# Load data
@st.cache_data
def load_data():
    with open('data/data_cleaned.pickle', 'rb') as f:
        df = pickle.load(f)
    return df

# Load the data
df = load_data()

# Sidebar
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
        - **Total Records:** {len(df):,}
        - **Features:** {len(df.columns)}
        - **Countries:** {df['Country'].nunique()}
        - **Platforms:** {df['Platform'].nunique()}
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
    st.markdown(
        """
        **Pre-trained Models:**
        - **Conflicts:** CatBoost Multiclass Classifier
        - **Addiction:** CatBoost Regressor with Rounding
        
        Models include full preprocessing pipelines - no manual encoding required!
        """
    )

    st.markdown("### 📁 Dataset Source")
    st.markdown(
        """
        Student Social Media Addiction Dataset  
        Contains comprehensive data on social media usage patterns, 
        academic performance, and mental health metrics.
        """
    )

# Main title
st.title("📱 SocialSphere Analytics: Social Media Conflicts & Addiction Prediction")

# Tabs
tab1, tab2 = st.tabs(["📊 Exploratory Data Analysis (EDA)", "🔮 Prediction Models"])

# -----------------------
# EDA TAB
# -----------------------
with tab1:
    st.header("📊 Exploratory Data Analysis")
    
    # Data overview
    st.subheader("📋 Dataset Overview")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(df.head(100))
    
    with col2:
        st.markdown("**Dataset Info:**")
        st.write(f"Shape: {df.shape}")
        st.write(f"Missing values: {df.isnull().sum().sum()}")
        
        st.markdown("**Numeric Columns:**")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        st.write(", ".join(numeric_cols))
        
        st.markdown("**Categorical Columns:**")
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        st.write(", ".join(categorical_cols))

    # Descriptive statistics
    st.subheader("📈 Descriptive Statistics")
    st.dataframe(df.describe())

    # Single horizontal box plot for all numeric features
    st.subheader("📊 Distribution of Numeric Features")
    numeric_features = ['Age', 'Daily_Usage', 'Sleep_Hrs', 'Mental_Health']

    # Prepare data in long format for a single box plot
    df_long = df[numeric_features].melt(var_name='Feature', value_name='Value')

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

   # Count plots of countries and platforms
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌍 Top Countries")
        country_counts = df['Country'].value_counts().head(11)
        fig = px.bar(x=country_counts.values, y=country_counts.index, 
                     orientation='h', title=f"Top 11 Countries out of {len(df['Country'].unique())}")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📱 Social Media Platforms")
        platform_counts = df['Platform'].value_counts()
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

    # Distributions of target variables in two columns
    st.subheader("🎯 Distributions of Target Variables")

    col1, col2 = st.columns(2)

    with col1:
        # st.markdown("**Conflicts Over Social Media**")
        fig_conflicts = px.histogram(
            df, 
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
        # st.markdown("**Addicted Score**")
        fig_addicted = px.histogram(
            df,
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


    # Distribution of conflicts by gender
    st.subheader("📊 Distribution of Conflicts by Gender")
    fig = px.box(df, x='Gender', y='Conflicts', 
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
    fig = px.box(df, x='Academic_Level', y='Addicted_Score', 
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
        top_countries = df['Country'].value_counts().head(11).index
        df_top_countries = df[df['Country'].isin(top_countries)]
        fig = px.box(df_top_countries, x='Country', y='Conflicts',
                     title="Conflicts by Top Countries",
                     color='Country',
                     color_discrete_sequence=px.colors.qualitative.Dark24)
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📱 Addicted Score by Platform")
        fig = px.box(df, x='Platform', y='Addicted_Score',
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


    # Correlation matrix
    st.subheader("🔗 Correlation Matrix of Numeric Features")
    numeric_df = df.select_dtypes(include=[np.number])
    
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

    # Box plot of conflicts based on relationship status
    st.subheader("💕 Conflicts by Relationship Status")
    fig = px.box(df, x='Relationship_Status', y='Conflicts',
                 title="Conflicts Over Social Media by Relationship Status",
                 color='Relationship_Status',
                 color_discrete_sequence=px.colors.qualitative.Alphabet)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Insights:**
    - All three relationship status groups have a similar median number of conflicts (around 3).
    - In Relationship and Complicated groups have a slightly narrower range, showing fewer students experience no conflicts in these groups.
    """)


# -----------------------
# PREDICTION TAB
# -----------------------
with tab2:
    st.header("🔮 Prediction Models")
    
    # Load MLflow models
    @st.cache_resource
    def load_conflicts_model():
        """Load the conflicts classification model from MLflow registry"""
        try:
            # Load the best conflict classification model
            logged_model = 'runs:/a7f3a1fd156443e58e7554ac1e8b53fa/model'
            model = mlflow.pyfunc.load_model(logged_model)
            return model
        except Exception as e:
            st.error(f"Error loading conflicts model: {e}")
            return None
    
    @st.cache_resource
    def load_addiction_model():
        """Load the addiction score regression model from MLflow registry"""
        try:
            # Try the simpler addiction model first
            model = mlflow.pyfunc.load_model("runs:/594b916daee046ff8f9fa0ed3aed8748/model")
            return model
        except Exception as e:
           st.error(f"Error loading addiction model: {e}")
           return None
    
    # Load models
    conflicts_model = load_conflicts_model()
    addiction_model = load_addiction_model()
    
    # Display model loading status
    st.markdown("### 🔧 Model Status")
    col1, col2 = st.columns(2)
    
    with col1:
        if conflicts_model is not None:
            st.success("✅ Conflicts Model: Loaded Successfully")
        else:
            st.error("❌ Conflicts Model: Failed to Load")
    
    with col2:
        if addiction_model is not None:
            st.success("✅ Addiction Model: Loaded Successfully")
        else:
            st.error("❌ Addiction Model: Failed to Load")
    
    if conflicts_model is None and addiction_model is None:
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
    
    # Model selection
    model_type = st.selectbox("Select Prediction Model:", 
                             ["Conflicts Prediction", "Addiction Score Prediction"])
    
    if model_type == "Conflicts Prediction":
        st.subheader("🔮 Predict Conflicts Over Social Media")
        
        if conflicts_model is None:
            st.error("❌ Conflicts model could not be loaded. Please check MLflow setup.")
        else:
            # User input form
            st.markdown("### 📝 Enter User Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.slider("Age", min_value=16, max_value=30, value=20)
                gender = st.selectbox("Gender", options=df['Gender'].unique())
                academic_level = st.selectbox("Academic Level", options=df['Academic_Level'].unique())
                country = st.selectbox("Country", options=df['Country'].unique())
                avg_daily_usage = st.slider("Daily Usage (Hours)", min_value=1.0, max_value=10.0, value=5.0, step=0.1)
            
            with col2:
                platform = st.selectbox("Most Used Platform", options=df['Platform'].unique())
                affects_academic = st.selectbox("Affects Academic Performance", options=df['Academic_Affects'].unique())
                sleep_hours = st.slider("Sleep Hours Per Night", min_value=3.0, max_value=10.0, value=7.0, step=0.1)
                mental_health = st.slider("Mental Health Score", min_value=1, max_value=10, value=7)
                relationship_status = st.selectbox("Relationship Status", options=df['Relationship_Status'].unique())
            
            # Make prediction
            if st.button("🔮 Predict Conflicts"):
                # Create input DataFrame with raw values (no preprocessing needed)
                input_df = pd.DataFrame([{
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
                
                # Run prediction directly
                preds = conflicts_model.predict(input_df)
                
                # Display results
                st.success(f"🎯 **Predicted Conflicts:** {preds[0]:.1f}")
                
                # Conflict level interpretation
                if preds[0] < 2:
                    conflict_level = "Low"
                    color = "green"
                elif preds[0] < 4:
                    conflict_level = "Moderate"
                    color = "orange"
                else:
                    conflict_level = "High"
                    color = "red"
                
                st.markdown(f"**Conflict Level:** :{color}[{conflict_level}]")
    
    else:  # Addiction Score Prediction
        st.subheader("🔮 Predict Addiction Score")
        
        if addiction_model is None:
            st.error("❌ Addiction model could not be loaded. Please check MLflow setup.")
        else:
            # User input form
            st.markdown("### 📝 Enter User Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.slider("Age", min_value=16, max_value=30, value=20, key="addiction_age")
                gender = st.selectbox("Gender", options=df['Gender'].unique(), key="addiction_gender")
                academic_level = st.selectbox("Academic Level", options=df['Academic_Level'].unique(), key="addiction_academic")
                country = st.selectbox("Country", options=df['Country'].unique(), key="addiction_country")
                avg_daily_usage = st.slider("Daily Usage (Hours)", min_value=1.0, max_value=10.0, value=5.0, step=0.1, key="addiction_usage")
            
            with col2:
                platform = st.selectbox("Most Used Platform", options=df['Platform'].unique(), key="addiction_platform")
                affects_academic = st.selectbox("Affects Academic Performance", options=df['Academic_Affects'].unique(), key="addiction_affects")
                sleep_hours = st.slider("Sleep Hours Per Night", min_value=3.0, max_value=10.0, value=7.0, step=0.1, key="addiction_sleep")
                mental_health = st.slider("Mental Health Score", min_value=1, max_value=10, value=7, key="addiction_mental")
                relationship_status = st.selectbox("Relationship Status", options=df['Relationship_Status'].unique(), key="addiction_relationship")
                conflicts = st.slider("Conflicts Over Social Media", min_value=0, max_value=5, value=2, key="addiction_conflicts")
            
            # Make prediction
            if st.button("🔮 Predict Addiction Score"):
                # Create input DataFrame with raw values (no preprocessing needed)
                input_df = pd.DataFrame([{
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
                
                # Run prediction directly
                preds = addiction_model.predict(input_df)
                
                # Display results
                st.success(f"🎯 **Predicted Addiction Score:** {preds[0]:.1f}/10")
                
                # Addiction level interpretation
                if preds[0] < 3:
                    addiction_level = "Low"
                    color = "green"
                elif preds[0] < 6:
                    addiction_level = "Moderate"
                    color = "orange"
                else:
                    addiction_level = "High"
                    color = "red"
                
                st.markdown(f"**Addiction Level:** :{color}[{addiction_level}]")
    
    # Feature importance section
    st.subheader("🎯 Model Information")
    st.markdown("""
    **MLflow Models Used:**
    - **Conflicts Prediction:** `conflict_catboost_multiclass` (CatBoost Classifier)
    - **Addiction Score Prediction:** `addicted_score_catboost_all_features+rounded` (CatBoost Regressor)
    
    These models are loaded directly from the MLflow model registry and include all necessary preprocessing steps.
    No manual feature engineering or encoding is required - just provide the raw input values!
    """)
