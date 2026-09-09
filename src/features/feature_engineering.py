import numpy as np
import os
import sys
import pandas as pd

from src.logger import configure_logger
from src.data.data_ingestion import load_data
from src.utils.schema_loader import read_yaml
from src.data.data_validation import validate_data
from src.data.data_preprocessing import DataPreprocessing, start_data_preprocessing
from src.exception import MyException
from config.constant import SCHEMA_PATH


logging = configure_logger()

class Feature_Engineering:
    def __init__(self, df: pd.DataFrame):
        self.shift_data = df
        logging.info("feature engineering intialized")

# feature engineering 
    def Engineer_features(self):
        try:
            logging.info("starting feature engineering...")
        # paring data
            self.shift_data['start_time'] = pd.to_datetime(self.shift_data['start_time'])
            self.shift_data['end_time'] = pd.to_datetime(self.shift_data['end_time'])

            # fixing overnight shifts where end time is less than start time
            mask = self.shift_data['end_time'] < self.shift_data['start_time']
            self.shift_data.loc[mask, 'end_time'] =self.shift_data.loc[mask, 'end_time'] + pd.Timedelta(days=1)

            # 1. shift duration in hours
            self.shift_data['shift_duration'] = (self.shift_data['end_time'] - self.shift_data['start_time']).dt.total_seconds() / 3600

            # defect rate calculation
            self.shift_data['defect_rate'] = self.shift_data['defect_count'] / self.shift_data['units_produced']

            # downtime ratio calculation
            self.shift_data['downtime_ratio'] = self.shift_data['downtime_minutes'] / (self.shift_data['shift_duration'] * 60)

            # temporal feature features
            self.shift_data['day_of_week'] = self.shift_data['date'].dt.dayofweek
            self.shift_data['hour_of_day'] = self.shift_data['start_time'].dt.hour

            logging.info(self.shift_data.head())
            return self.shift_data
        except Exception as e:
            logging.error(f"error occured while engineering new features {e}")
            raise MyException(e, sys)
        
        # feature selection
    def Feature_Selection(self):
        try: 
            schema = read_yaml(SCHEMA_PATH)
            colums_to_drop = schema['columns']['columns_to_drop']

            existing_cols = [column for column in colums_to_drop if column in self.shift_data.columns]
            self.shift_data.drop(columns = existing_cols, inplace= True)

            processor = DataPreprocessing(self.shift_data)
            self.shift_data = processor.remove_duplicates()

            logging.info("Dropped unwanted columns and remove duplicates")
            logging.info(self.shift_data.head())

            return self.shift_data
        
        except Exception as e:
            logging.error(f"error occured during feature selection ")
            raise MyException(e, sys)

    def Feature_engineering_Engine(self):
        try: 
            self.shift_data = self.Engineer_features()
            self.shift_data = self.Feature_Selection()
            logging.info("feature engineering completed")
            return self.shift_data
        except Exception as e:
            raise MyException(e, sys)
        
def start_feature_engineering(shift_data: pd.DataFrame):
    try:
        engineer = Feature_Engineering(shift_data)
        shift_data = engineer.Feature_engineering_Engine()

        ## calling splitting processor 
        processor = DataPreprocessing(shift_data)
        X_train, X_test, y_train, y_test = processor.train_test_splitting(shift_data)

        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        logging.error("error occured during features engineering initialization...")
        raise MyException(e, sys)

shift_data = load_data()
validated_data = validate_data(shift_data)
processed_data = start_data_preprocessing(validated_data)
X_train, X_test, y_train, y_test = start_feature_engineering (processed_data)

print("feature engineering completed ....")
print(X_train.head())
print(X_test.head())

