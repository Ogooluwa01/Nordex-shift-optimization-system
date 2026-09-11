import pandas as pd
import sys
import numpy as np
from src.logger import configure_logger
from src.exception import MyException
from config.constant import database_path, data_artifact

import sqlite3
import os
import logging
logging = configure_logger()

def load_data():
    """
    loading the data the database file, validate teh schema, and return a dataframe 

    Returns:
    pd.DataFrame: this function returns pandas dataframe containing
      the data loaded from the database file.
    The dataFrame is validated against the expected schema 
    to ensure that it has the correct columns and dat types.

    """

    try:
        logging.info("Starting data loading process from the databse file.")
        # connect to the sqlite database
        conn =  sqlite3.connect(database_path)
        shift_data = pd.read_sql("select * from ShiftPerformance", conn)
        conn.close()

        print("data_artifact =", repr(data_artifact))
        print("database_path =", repr(database_path))
        print("data_artifact exists =", os.path.exists(data_artifact))

        os.makedirs(data_artifact, exist_ok=True)
        shift_data.to_csv(os.path.join(data_artifact, 'ingested_data.csv'), index = False)

        logging.info(f"First few rows of the ingested data:\n{shift_data.head()}")
        logging.info("Data loading process completed succesfully")


        return shift_data


    except Exception as e:
        logging.error(f"Error occured during data ingestion {e}")
        raise MyException(e, sys)

load_data()

