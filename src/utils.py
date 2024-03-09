import os
import sys
import pandas as pd
import numpy as np
import pickle
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from src.logger import logging

def save_object(file_path, obj):
    '''
        This method saves the object as a file.
    '''
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]
            
            random_search = RandomizedSearchCV(model, param, cv=3)
            random_search.fit(X_train, y_train)
            logging.info(f"Grid search completed for {list(models.keys())[i]}")
            
            model.set_params(**random_search.best_params_)
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
        
        return report
        
    except Exception as e:
        raise CustomException(e, sys)