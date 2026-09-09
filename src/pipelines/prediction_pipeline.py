import pandas as pd
import sys
import numpy as np
from src.logger import configure_logger
from src.exception import MyException

logging = configure_logger()
def prediction_pipeline(input_data: dict, model):
    """
    uses the preloaded model from mlflow to make predictions on the input data.

    Input:
        input_data: diction from the API 
        model: preloaded model from MLFLOW(passed from the api)
    Output:
        prediction(list)
    """
    try:
        logging.info("preparing input data for prediction...")
        # convert input data to dataframe
        df = pd.DataFrame([input_data])

        prediction = model.predict(df)
        logging.info(f"prediction completed: {predictions}")
        return prediction.tolist()
    except Exception as e:
        logging.error(f"error occured during prediction: {e}")
        raise MyException(e, sys)