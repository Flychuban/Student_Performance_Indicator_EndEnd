import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings

df = pd.read_csv('data/StudentsPerformance.csv')

# Add columns "Total Score" and "Average" to the data
df["total_score"] = df["math score"] + df["reading score"] + df["writing score"]
df["average"] = round(df["total_score"] / 3, 2)

# Split the data into features and target
y = df["average"]
X = df.drop(["average", "total_score"], axis=1)

# One-hot encode the categorical features and scale the numerical features
numeric_features = X.select_dtypes(exclude="object").columns
categorical_features = X.select_dtypes(include="object").columns

numeric_transformer = StandardScaler()
onehot_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    [
        ("OneHotEncoder", onehot_transformer, categorical_features),
        ("StandardScaler", numeric_transformer, numeric_features)
    ])

X = preprocessor.fit_transform(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train.shape