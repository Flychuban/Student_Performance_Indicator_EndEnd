import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_data():
    # Load the data
    df = pd.read_csv('data/StudentsPerformance.csv')
    return df
def data_check(df):
    # Checks how many columns have missing values
    df.isna().sum()
    
    # Checks how many duplicate rows are in the data
    df.duplicated().sum()
    
    df.info()
    
    # Checks for unique values in the data
    df.nunique()
    
    df.describe()
    
    numeric_features = [feature for feature in df.columns if df[feature].dtypes != 'O']
    categorical_features = [feature for feature in df.columns if df[feature].dtypes == 'O']
    
    # Add columns "Total Score" and "Average" to the data
    df["total_score"] = df["math score"] + df["reading score"] + df["writing score"]
    df["average"] = df["total_score"] / 3
    

df = get_data()
data_check(df)