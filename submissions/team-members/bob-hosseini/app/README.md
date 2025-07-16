# SocialSphere Analytics App

A comprehensive Streamlit application for analyzing social media usage patterns, conflicts, and addiction scores among students.

## Features

### 📊 Exploratory Data Analysis (EDA)
- **Distribution Analysis**: Conflicts by gender, addiction scores by academic level
- **Geographic Analysis**: Country and platform distributions
- **Statistical Visualizations**: Box plots, histograms, correlation matrices
- **Relationship Analysis**: Conflicts based on relationship status
- **Interactive Plots**: All visualizations are interactive using Plotly

### 🔮 Prediction Models
- **Conflicts Prediction**: Predict social media conflicts based on user characteristics
- **Addiction Score Prediction**: Forecast addiction scores (1-10 scale)
- **Feature Importance**: Understand which factors most influence predictions
- **Model Performance**: Real-time model evaluation metrics

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the App**:
   ```bash
   streamlit run app.py
   ```

3. **Access the App**:
   Open your browser and go to `http://localhost:8501`

## Data Source

The app uses the "Students Social Media Addiction" dataset which includes:
- **707 student records** from various countries
- **13 features** including demographic, usage, and behavioral data
- **Target variables**: Conflicts_Over_Social_Media (0-5) and Addicted_Score (1-10)

## Features Included

### Categorical Features
- Gender (Male/Female)
- Academic Level (High School/Undergraduate/Graduate)
- Country (50+ countries)
- Most Used Platform (Instagram, TikTok, Facebook, etc.)
- Affects Academic Performance (Yes/No)
- Relationship Status (Single/In Relationship/Complicated)

### Numerical Features
- Age (16-30)
- Average Daily Usage Hours (1-10)
- Sleep Hours Per Night (3-10)
- Mental Health Score (1-10)
- Conflicts Over Social Media (0-5)
- Addicted Score (1-10)

## Model Details

- **Algorithm**: Random Forest Regressor
- **Features**: All available features with proper encoding
- **Validation**: 80/20 train-test split
- **Metrics**: MAE, RMSE, R² Score
- **Caching**: Models are cached for performance

## Usage

1. **EDA Tab**: Explore the data through various interactive visualizations
2. **Prediction Tab**: 
   - Select prediction type (Conflicts or Addiction Score)
   - Fill in user information using dropdowns and sliders
   - Click "Predict" to get results
   - View model performance and feature importance

## Technical Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn
- **Caching**: Streamlit cache decorators

## Project Structure

```
app/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Contributing

Feel free to contribute by:
- Adding new visualizations
- Improving model performance
- Enhancing the UI/UX
- Adding new features

## License

This project is part of the SuperDataScience community initiative. 