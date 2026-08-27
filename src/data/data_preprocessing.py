from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sys
from config.constant import target_column

from src.logger import configure_logger
from src.exception import MyException

logging = configure_logger()

class DataPreprocessing:
    def __int__(self, df: pd.DataFrame):
        self.shift_data = df
        logging.info("Data preprocessing initialized...")

    def filling_missing_values(self):
        try:
            # filing the missing values
            self.shift_data["date"] = pd.to_datetime(shift_data["date"])

            self.shift_data=self.shift_data.sort_values(by="date")

                #fil ling the missing values for temperature and humidity using forward fill method and then filling any remaining missing values with the mean of the respective columns
            self.shift_data["temperature"]=self.shift_data["temperature"].fillna(method="ffill").fillna(shift_data['temperature'].mean())
            self.shift_data["humidity"]=shift_data["humidity"].fillna(method="ffill").fillna(shift_data['humidity'].mean())

                # fill the timestamp
            self.shift_data["timestamp"]=self.shift_data["timestamp"].fillna(method="ffill") 
                    # fill the categorical mainatainenace field

            self.shift_data['issue_type'] =self.shift_data["issue_type"].fillna("No Issue")
            self.shift_data['maintenance_downtime'] =self.shift_data["maintenance_downtime"].fillna(0)
            self.shift_data['resolved_by'] =self.shift_data["resolved_by"].fillna("No Issue resolved")


            self.shift_data =self.shift_data.drop(columns=["maintenance_id"])

            self.shift_data.isna().sum()

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
            X = shift_data.drop
            # i would continue from here i had to go to work
