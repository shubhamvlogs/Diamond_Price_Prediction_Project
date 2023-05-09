from sklearn.impute import SimpleImputer # Handling Missing Values
from sklearn.preprocessing import StandardScaler #  Handling Features Scaling
from sklearn.preprocessing import OrdinalEncoder #Ordianl Encoding

## Pipelines

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import sys,os
from dataclasses import dataclass
import pandas as pd
import numpy as np


from src.logger import CustomException
from src.logger import logging
from src.utils  import save_object


## Data Transformation config

@dataclass
class DataTransformationconfig:
    preprocessor_ob_file_path=os.path.join('artifacts','preprocessor.pkl')
    
    
    
## Data Ingestion class

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformation()
        
        
    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
            
            # Define which columns should be ordinal-encoded and which should be scaled 
            categorical_cols= [ 'cut','color','clarity']
            numerical_cols = ['carat','depth','table','x','y','z']
            
            # Define the custom ranking for each ordinal variable
            cut_categories=['Fair','Good','Very Good','Premium','Ideal']
            color_categories=['D','E','F','G','H','I','J']
            clarity_categories=['I1','SI2','SI1','VS2','VS1','VVS2','VVS1',
                'IF' ]
            
            logging.info('Pipeline Initiated')
            
            
            ## Numerical Pipeline
            
            

            num_pipeline=Pipeline(
                        steps=[
                        ('imputer', SimpleImputer(strategy="median")),
                        ('scaler',StandardScaler())
    
    
        
                        ]
                    )   
            
            ## Categorical Pipeline
            cat_pipeline = Pipeline(
                        steps=[
                        ('imputer',SimpleImputer(strategy='most_frequent')),
                        ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                        ('scaler',StandardScaler())
                        ]
                    )
            
            preprocessor=ColumnTransformer([
    
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])
            
            return preprocessor
        
            logging.info('Pipeline Completed')
            
            
        except Exception as e:
            
            
            logging.info('Error in Data Transformation')
            raise CustomException(e,sys)
               
        
        
    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            # Reading train and test data
            train_df=pd.read_csv(train_data_path)
            test_df=pd.read_csv(test_data_path)
            
            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head: \n {train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head: \n {test_df.head().to_string()}')
            
            
            logging.info('Obtaining preprocessing object')
            
            preprocessing_obj = self.get_data_transformation_object()
            
            
            target_column_name = 'price'
            drop_columns = [target_column_name,'id']
            
            
            ## features dividing into independent and dependent features
            
            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            ## apply the transformation
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            
            logging.info("Applying preprocessing object on training and testing datasets")
            
            
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                
                obj=preprocessing_obj
            )
            
            logging.info('Processor pickle is created and saved')
            
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
                
            )
            
        
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")
            
            raise CustomException(e,sys)
            
            
            
            
        
        


