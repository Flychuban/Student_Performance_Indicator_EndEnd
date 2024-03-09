import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline

from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

import pickle

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "trained_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("Starting training and test input data")
            X_train, y_train, X_test, y_test = (train_array[:,:-1], train_array[:, -1], test_array[:, :-1], test_array[:, -1])
            
            models = {
                "RandomForestRegressor": RandomForestRegressor(),
                "GradientBoostingClassifier": GradientBoostingClassifier(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "LogisticRegression": LogisticRegression(),
                "SVR": SVR(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "XGBRegressor": XGBRegressor(),
                "LinearRegression": LinearRegression(),
                "CatBoostRegressor": CatBoostRegressor()
            }
            
            params = {
                "DecisionTreeRegressor": {
                    'criterion': ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                },
                "RandomForestRegressor": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                "GradientBoostingClassifier": {
                    'learning_rate': [0.01, 0.1, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "LinearRegression": {},
                "KNeighborsRegressor": {
                    'n_neighbors': [5, 7, 9, 11]
                },
                "XGBRegressor": {
                    'learning_rate': [0.01, 0.1, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoostRegressor": {
                    'learning_rate': [0.01, 0.1, 0.05],
                    'depth': [6, 8, 10],
                    'iterations': [30, 50, 100]
                },
                "AdaBoostRegressor": {
                    'learning_rate': [0.01, 0.1, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "LogisticRegression": {
                    'C': [0.01, 0.1, 1, 10, 100, 1000]
                },
                "SVR": {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
                }                
            }
            
            model_report:dict = evaluate_model(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, models = models, params = params)
            
            # Get the best model score from the report
            best_model_score = max(sorted(model_report.values()))
            
            # Get best model name from the report
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("Best model score is less than 0.6", sys)
            
            logging.info(f"Best model name: {best_model_name} found with score: {best_model_score}")
            
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predictions = best_model.predict(X_test)
            prediction_r2_score = r2_score(y_test, predictions)
            logging.info(f"Predicting R2 score: {prediction_r2_score}")
            
            return prediction_r2_score
        
        except Exception as e:
            raise CustomException(e, sys)