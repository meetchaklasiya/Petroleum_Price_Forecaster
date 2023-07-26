import sys
import pandas as pd
from src.exception import CustomException
from src.utlis import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            model_path=os.path.join('artifacts','model.pkl')
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            # data_scaled=preprocessor.transform(features)
            preds=model.predict(features)
            return preds
        except Exception as e:
            raise CustomException(e,sys)
    
class CustomData:
    def __init__(self,
        date:int,
        month:int,
        year:int
                ):
        self.date = date
        
        self.month = month
        
        self.year = year
    
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'date': [self.date],
                'month': [self.month],
                'year': [self.year]
            }
            
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)