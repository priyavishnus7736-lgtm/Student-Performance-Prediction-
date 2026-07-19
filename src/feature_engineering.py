import pandas as pd

def engineer_student_features(df):
    df = df.copy()
    if all(col in df.columns for col in ['Assignments Score', 'Projects Score', 'Test Score']):
        df['Total_Assessment_Score'] = df['Assignments Score'] + df['Projects Score'] + df['Test Score']
    if 'Study Hours' in df.columns and 'Attendance (%)' in df.columns:
        df['Attendance_to_Study_Ratio'] = df['Attendance (%)'] / (df['Study Hours'] + 1e-5)
    if 'Study Hours' in df.columns and 'Assignments Score' in df.columns:
        df['Study_Efficiency'] = df['Study Hours'] * df['Assignments Score']
    return df
