import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    if 'Attendance (%)' in df.columns:
        df['Attendance (%)'] = df['Attendance (%)'].clip(0, 100)
    return df

def preprocess_features(df):
    df = df.copy()
    le = LabelEncoder()
    categorical_cols = ['Gender', 'Participation', 'Behavior', 'Extracurricular Activities']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
    return df
