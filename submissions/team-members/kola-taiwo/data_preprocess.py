# src/preprocessing.py
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(df):
    # Drop Student ID if not required
    df = df.drop(columns=['Student ID'])
    
    # Encode binary categorical variables
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    df['Affects_Academic_Performance'] = df['Affects_Academic_Performance'].map({'Yes': 1, 'No': 0})
    
    # List of categorical variables for one-hot encoding
    categorical_vars = ['Academic_Level', 'Country', 'Most_Used_Platform', 'Relationship_Status']
    
    # Numerical variables
    numerical_vars = [
        'Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Mental_Health_Score'
    ]
    
    # Define transformers
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_vars),
            ('cat', categorical_transformer, categorical_vars)
        ])
    
    X = df.drop(columns=['Conflicts_Over_Social_Media', 'Addicted_Score'])
    y_conflict = df['Conflicts_Over_Social_Media']
    y_addiction = df['Addicted_Score']
    
    X_processed = preprocessor.fit_transform(X)
    
    return X_processed, y_conflict, y_addiction
