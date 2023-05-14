import os
import sys
import pickle
import pandas as pd
import numpy as np


from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score, mean_absolute_error





def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        

    except Exception as e:
        raise CustomException(e, sys)
    
    
logging.info(" i am creating here evaluate model")



def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            
            # Train model
            model.fit(X_train, y_train)

            # Predict Testing data
            y_test_pred = model.predict(X_test)

            # Get R2 scires for train and test data
            # train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score
            


        return report
    
        
    
    except Exception as e:
        
        raise CustomException(e, sys)
    
#logging.critical("here i am handling error ")
