from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sys
from config.constant import target_column
from sklearn.model_selection import train_test_split
from src.logger import configure_logger
from src.data.data_ingestion import load_data
from src.data.data_validation import validate_data
from src.exception import MyException

logging = configure_logger()

class DataPreprocessing:
    def __init__(self, df: pd.DataFrame):
        self.shift_data = df
        logging.info("Data preprocessing initialized...")

    def filling_missing_values(self):
        try:
            # filing the missing values
            self.shift_data["date"] = pd.to_datetime(self.shift_data["date"])

            self.shift_data=self.shift_data.sort_values(by="date")

                #fil ling the missing values for temperature and humidity using forward fill method and then filling any remaining missing values with the mean of the respective columns
            self.shift_data["temperature"]=self.shift_data["temperature"].fillna(method="ffill").fillna(self.shift_data['temperature'].mean())
            self.shift_data["humidity"]=self.shift_data["humidity"].fillna(method="ffill").fillna(self.shift_data['humidity'].mean())

                # fill the timestamp
            self.shift_data["timestamp"]=self.shift_data["timestamp"].fillna(method="ffill") 
                    # fill the categorical mainatainenace field

            self.shift_data['issue_type'] =self.shift_data["issue_type"].fillna("No Issue")
            self.shift_data['maintenance_downtime'] =self.shift_data["maintenance_downtime"].fillna(0)
            self.shift_data['resolved_by'] =self.shift_data["resolved_by"].fillna("No Issue resolved")


            self.shift_data =self.shift_data.drop(columns=["maintenance_id"])

            logging.info(self.shift_data.isna().sum())

            return self.shift_data
        
        except Exception as e:
            logging.error(f"error occured when filling missing value {e}")
            raise MyException(e, sys)

    def remove_duplicates(self):
        try:
            duplicates = self.shift_data.duplicated().sum()

            if duplicates > 0:
                logging.warning(f"removing {duplicates} duplicated rows")
                self.shift_data.drop_duplicates(inplace= True)

            return self.shift_data
            
        except Exception as e:
            logging.error(f"error occured when dropping duplicated values {e}")
            raise MyException(e, sys)
        
    def preprocess_data(self):
        try:
            logging.info("starting the data preprocessing pipeline ...")
            self.shift_data = self.filling_missing_values()
            self.shift_data = self.remove_duplicates()
            logging.info("data preprocessing completed.")
            return self.shift_data
            
        
        except Exception as e:
            logging.error(f"error occured during data preprocessing {e}")
            raise MyException(e, sys)

    def split_X_y(self, shift_data:pd.DataFrame):
        try: 
            X = shift_data.drop(columns=[target_column], axis=1)
            y = shift_data[target_column]

            return X, y
        
        except Exception as e:
            logging.error(f"error occured during data spilting {e}")
            raise MyException(e, sys)

    def train_test_splitting(self, shift_data: pd.DataFrame):
        try:
            X, y = self.split_X_y(shift_data)
            X_train, X_test, y_train, y_test = train_test_split ( X, y, test_size=0.2, random_state=42 )
            logging.info("Training and testing split completed")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("error occured while splitting data into training and testing {e}")
            raise MyException(e, sys)



def start_data_preprocessing(shift_data: pd.DataFrame):
    try:
        processor = DataPreprocessing(shift_data)
        shift_data = processor.preprocess_data()
        return shift_data
    except Exception as e:
        logging.error(f"error occured during data preprocessing....")
        raise MyException(e, sys)
    

shift_data = load_data()
shift_data = validate_data(shift_data)
start_data_preprocessing(shift_data=shift_data)
            
