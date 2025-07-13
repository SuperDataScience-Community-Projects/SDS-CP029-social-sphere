import streamlit as st
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
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="SocialSphere Analytics",
    layout="wide",
    initial_sidebar_state="auto"
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("submissions/team-members/aditi-phadnis/Students Social Media Addiction.csv")
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
        - **Platforms:** {df['Most_Used_Platform'].nunique()}
        """
    )

    st.markdown("### 🎯 Target Variables")
    st.markdown(
        """
        - **Conflicts_Over_Social_Media:** Number of conflicts (0-5)
        - **Addicted_Score:** Addiction level (1-10)
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
    numeric_features = ['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Mental_Health_Score']

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
        platform_counts = df['Most_Used_Platform'].value_counts()
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
            x='Conflicts_Over_Social_Media', 
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
    fig = px.box(df, x='Gender', y='Conflicts_Over_Social_Media', 
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
        fig = px.box(df_top_countries, x='Country', y='Conflicts_Over_Social_Media',
                     title="Conflicts by Top Countries",
                     color='Country',
                     color_discrete_sequence=px.colors.qualitative.Dark24)
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📱 Addicted Score by Platform")
        fig = px.box(df, x='Most_Used_Platform', y='Addicted_Score',
                     title="Addicted Score by Social Media Platform",
                     color='Most_Used_Platform',
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
    - **Conflicts_Over_Social_Media** and **Addicted_Score** both show strong correlations with several key features:
        - **Avg_Daily_Usage_Hours**: Positive correlation (higher usage is linked to more conflicts and higher addiction scores)
        - **Mental_Health_Score**: Negative correlation (lower mental health scores are associated with higher conflicts and addiction)
        - **Sleep_Hours_Per_Night**: Negative correlation (less sleep is linked to more conflicts and higher addiction)
    """)

    # Box plot of conflicts based on relationship status
    st.subheader("💕 Conflicts by Relationship Status")
    fig = px.box(df, x='Relationship_Status', y='Conflicts_Over_Social_Media',
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
    
    # Prepare data for modeling
    @st.cache_data
    def prepare_model_data():
        # Create a copy for modeling
        df_model = df.copy()
        
        # Encode categorical variables
        le_gender = LabelEncoder()
        le_academic = LabelEncoder()
        le_country = LabelEncoder()
        le_platform = LabelEncoder()
        le_affects = LabelEncoder()
        le_relationship = LabelEncoder()
        
        df_model['Gender_Encoded'] = le_gender.fit_transform(df_model['Gender'])
        df_model['Academic_Level_Encoded'] = le_academic.fit_transform(df_model['Academic_Level'])
        df_model['Country_Encoded'] = le_country.fit_transform(df_model['Country'])
        df_model['Most_Used_Platform_Encoded'] = le_platform.fit_transform(df_model['Most_Used_Platform'])
        df_model['Affects_Academic_Performance_Encoded'] = le_affects.fit_transform(df_model['Affects_Academic_Performance'])
        df_model['Relationship_Status_Encoded'] = le_relationship.fit_transform(df_model['Relationship_Status'])
        
        return df_model, {
            'gender': le_gender,
            'academic': le_academic,
            'country': le_country,
            'platform': le_platform,
            'affects': le_affects,
            'relationship': le_relationship
        }
    
    df_model, label_encoders = prepare_model_data()
    
    # Model selection
    model_type = st.selectbox("Select Prediction Model:", 
                             ["Conflicts Prediction", "Addiction Score Prediction"])
    
    if model_type == "Conflicts Prediction":
        st.subheader("🔮 Predict Conflicts Over Social Media")
        
        # Feature selection for conflicts prediction
        features_conflicts = ['Age', 'Gender_Encoded', 'Academic_Level_Encoded', 'Country_Encoded',
                             'Avg_Daily_Usage_Hours', 'Most_Used_Platform_Encoded', 
                             'Affects_Academic_Performance_Encoded', 'Sleep_Hours_Per_Night',
                             'Mental_Health_Score', 'Relationship_Status_Encoded']
        
        X_conflicts = df_model[features_conflicts]
        y_conflicts = df_model['Conflicts_Over_Social_Media']
        
        # Train model
        @st.cache_resource
        def train_conflicts_model():
            X_train, X_test, y_train, y_test = train_test_split(X_conflicts, y_conflicts, 
                                                               test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            return model, X_test, y_test
        
        conflicts_model, X_test_conflicts, y_test_conflicts = train_conflicts_model()
        
        # User input form
        st.markdown("### 📝 Enter User Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", min_value=16, max_value=30, value=20)
            gender = st.selectbox("Gender", options=df['Gender'].unique())
            academic_level = st.selectbox("Academic Level", options=df['Academic_Level'].unique())
            country = st.selectbox("Country", options=df['Country'].unique())
            avg_daily_usage = st.slider("Average Daily Usage (Hours)", min_value=1.0, max_value=10.0, value=5.0, step=0.1)
        
        with col2:
            platform = st.selectbox("Most Used Platform", options=df['Most_Used_Platform'].unique())
            affects_academic = st.selectbox("Affects Academic Performance", options=df['Affects_Academic_Performance'].unique())
            sleep_hours = st.slider("Sleep Hours Per Night", min_value=3.0, max_value=10.0, value=7.0, step=0.1)
            mental_health = st.slider("Mental Health Score", min_value=1, max_value=10, value=7)
            relationship_status = st.selectbox("Relationship Status", options=df['Relationship_Status'].unique())
        
        # Make prediction
        if st.button("🔮 Predict Conflicts"):
            # Encode user inputs
            gender_encoded = label_encoders['gender'].transform([gender])[0]
            academic_encoded = label_encoders['academic'].transform([academic_level])[0]
            country_encoded = label_encoders['country'].transform([country])[0]
            platform_encoded = label_encoders['platform'].transform([platform])[0]
            affects_encoded = label_encoders['affects'].transform([affects_academic])[0]
            relationship_encoded = label_encoders['relationship'].transform([relationship_status])[0]
            
            # Create input array
            user_input = np.array([age, gender_encoded, academic_encoded, country_encoded,
                                  avg_daily_usage, platform_encoded, affects_encoded,
                                  sleep_hours, mental_health, relationship_encoded]).reshape(1, -1)
            
            # Predict
            prediction = conflicts_model.predict(user_input)[0]
            
            # Display results
            st.success(f"🎯 **Predicted Conflicts:** {prediction:.1f}")
            
            # Model performance
            y_pred_test = conflicts_model.predict(X_test_conflicts)
            mae = mean_absolute_error(y_test_conflicts, y_pred_test)
            rmse = np.sqrt(mean_squared_error(y_test_conflicts, y_pred_test))
            r2 = r2_score(y_test_conflicts, y_pred_test)
            
            st.markdown("### 📊 Model Performance")
            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", f"{mae:.2f}")
            col2.metric("RMSE", f"{rmse:.2f}")
            col3.metric("R² Score", f"{r2:.3f}")
    
    else:  # Addiction Score Prediction
        st.subheader("🔮 Predict Addiction Score")
        
        # Feature selection for addiction prediction
        features_addiction = ['Age', 'Gender_Encoded', 'Academic_Level_Encoded', 'Country_Encoded',
                             'Avg_Daily_Usage_Hours', 'Most_Used_Platform_Encoded', 
                             'Affects_Academic_Performance_Encoded', 'Sleep_Hours_Per_Night',
                             'Mental_Health_Score', 'Relationship_Status_Encoded', 'Conflicts_Over_Social_Media']
        
        X_addiction = df_model[features_addiction]
        y_addiction = df_model['Addicted_Score']
        
        # Train model
        @st.cache_resource
        def train_addiction_model():
            X_train, X_test, y_train, y_test = train_test_split(X_addiction, y_addiction, 
                                                               test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            return model, X_test, y_test
        
        addiction_model, X_test_addiction, y_test_addiction = train_addiction_model()
        
        # User input form
        st.markdown("### 📝 Enter User Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", min_value=16, max_value=30, value=20, key="addiction_age")
            gender = st.selectbox("Gender", options=df['Gender'].unique(), key="addiction_gender")
            academic_level = st.selectbox("Academic Level", options=df['Academic_Level'].unique(), key="addiction_academic")
            country = st.selectbox("Country", options=df['Country'].unique(), key="addiction_country")
            avg_daily_usage = st.slider("Average Daily Usage (Hours)", min_value=1.0, max_value=10.0, value=5.0, step=0.1, key="addiction_usage")
        
        with col2:
            platform = st.selectbox("Most Used Platform", options=df['Most_Used_Platform'].unique(), key="addiction_platform")
            affects_academic = st.selectbox("Affects Academic Performance", options=df['Affects_Academic_Performance'].unique(), key="addiction_affects")
            sleep_hours = st.slider("Sleep Hours Per Night", min_value=3.0, max_value=10.0, value=7.0, step=0.1, key="addiction_sleep")
            mental_health = st.slider("Mental Health Score", min_value=1, max_value=10, value=7, key="addiction_mental")
            relationship_status = st.selectbox("Relationship Status", options=df['Relationship_Status'].unique(), key="addiction_relationship")
            conflicts = st.slider("Conflicts Over Social Media", min_value=0, max_value=5, value=2, key="addiction_conflicts")
        
        # Make prediction
        if st.button("🔮 Predict Addiction Score"):
            # Encode user inputs
            gender_encoded = label_encoders['gender'].transform([gender])[0]
            academic_encoded = label_encoders['academic'].transform([academic_level])[0]
            country_encoded = label_encoders['country'].transform([country])[0]
            platform_encoded = label_encoders['platform'].transform([platform])[0]
            affects_encoded = label_encoders['affects'].transform([affects_academic])[0]
            relationship_encoded = label_encoders['relationship'].transform([relationship_status])[0]
            
            # Create input array
            user_input = np.array([age, gender_encoded, academic_encoded, country_encoded,
                                  avg_daily_usage, platform_encoded, affects_encoded,
                                  sleep_hours, mental_health, relationship_encoded, conflicts]).reshape(1, -1)
            
            # Predict
            prediction = addiction_model.predict(user_input)[0]
            
            # Display results
            st.success(f"🎯 **Predicted Addiction Score:** {prediction:.1f}/10")
            
            # Addiction level interpretation
            if prediction < 3:
                addiction_level = "Low"
                color = "green"
            elif prediction < 6:
                addiction_level = "Moderate"
                color = "orange"
            else:
                addiction_level = "High"
                color = "red"
            
            st.markdown(f"**Addiction Level:** :{color}[{addiction_level}]")
            
            # Model performance
            y_pred_test = addiction_model.predict(X_test_addiction)
            mae = mean_absolute_error(y_test_addiction, y_pred_test)
            rmse = np.sqrt(mean_squared_error(y_test_addiction, y_pred_test))
            r2 = r2_score(y_test_addiction, y_pred_test)
            
            st.markdown("### 📊 Model Performance")
            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", f"{mae:.2f}")
            col2.metric("RMSE", f"{rmse:.2f}")
            col3.metric("R² Score", f"{r2:.3f}")
    
    # Feature importance
    st.subheader("🎯 Feature Importance")
    if model_type == "Conflicts Prediction":
        feature_importance = pd.DataFrame({
            'Feature': features_conflicts,
            'Importance': conflicts_model.feature_importances_
        }).sort_values('Importance', ascending=True)
    else:
        feature_importance = pd.DataFrame({
            'Feature': features_addiction,
            'Importance': addiction_model.feature_importances_
        }).sort_values('Importance', ascending=True)
    
    fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h',
                 title=f"Feature Importance for {model_type}")
    st.plotly_chart(fig, use_container_width=True)
