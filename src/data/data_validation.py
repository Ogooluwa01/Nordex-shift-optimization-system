from src.logger import configure_logger
from src.exception import MyException
import pandas as pd
import sys
import os

from src.data.data_ingestion import load_data

logging = configure_logger()

class DataValidation:
    """
    This class is responsible for validating the data loaded from the database file.
    It checks if the data has the expected schema, including the correct columns and data types.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        logging.info("DataValidation class initialized successfully")

    def check_empty_data(self):
        """
        Checks if the dataframe is empty and raises an exception if it is .
        """
        logging.info("checking if the datadrame is empty")
        if self.df is None or self.df.empty:
            logging.error("The dataframe is empty. No data to validate")
            raise MyException("The dataframe is empty.No data to validate.")
        logging.info("The data frame is not empty. Proceeding with validation.")

    def checking_for_missing_values(self):
        logging.info("Checking for missing values in the dataframe")
        missing_value = self.df.isna().sum()

        if missing_value.sum() > 0:
            logging.warning(f"Missing values found in the dataframe: \n{missing_value}")
        else:
            logging.info("No missing values found in the dataframe.")
        return missing_value

    def checking_for_duplicates(self):
        logging.info("Checking for duplicate rows in the dataframe")
        duplicates = self.df.duplicated().sum()

        if duplicates > 0:
            logging.warning(f"Duplicates found in the dataframe: {duplicates} duplicate rows.")
        else:
            logging.info("No duplicate rows found in the dataframe.")
        return duplicates
    
def validate_data(df: pd.DataFrame):
    """
        This function validates the data loaded from teh database file.
        It checks if the data has the expected schema, including the correct columns and data types.

        Args:
            df (pd.DataFrame): The dataFrame to be validated.
        Returns:
            pd.DataFrame: The validated dataframe.
        """
    try:
        logging.info("Starting data validation process")
        validator = DataValidation(df)

        validator.check_empty_data()
        validator.checking_for_missing_values()
        validator.checking_for_duplicates()

        logging.info("Data Validation process completed successfully")

        return df
    except Exception as e:
        logging.error(f"Error occured during data validation: {e}")
        raise MyException(e, sys)


shift_data = load_data()
validate_data(shift_data)




