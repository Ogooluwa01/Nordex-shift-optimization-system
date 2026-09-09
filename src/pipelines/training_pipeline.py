import mlflow
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from src.data.data_ingestion import load_data
from src.data.data_validation import validate_data
from src.data.data_preprocessing import DataPreprocessing, start_data_preprocessing
from src.features.feature_engineering import Feature_Engineering, start_feature_engineering
import sys
from src.models.model_training import ModelTrainer
from src.models.model_tracking import ModelTracker
from src.logger import configure_logger
from src.exception import MyException

logging = configure_logger()

def start_model_training():
    try:
        shift_data = load_data()
        validated_data = validate_data(shift_data)
        processed_data = start_data_preprocessing(validated_data)
        X_train, X_test, y_train, y_test = start_feature_engineering(processed_data)
        print("feature engineering completed ....")
        logging.info("Initializing model training...")
        trainer = ModelTrainer(X_train, X_test, y_train, y_test)
        pipeline = trainer.train_model()
        r2, mae = trainer.evaluate_model()
        logging.info("model training completed.")
        model_tracker = ModelTracker()
        was_registered = model_tracker.push_model(
            model = pipeline,
            r2_score=r2, mae_score=mae)
        if was_registered:
            logging.info("New pipeline registered successfully to mlflow.")
        else:
            logging.info(f"Existing model has a better performance than the new pipeline ...")

            return pipeline, r2, mae

    except Exception as e:
        raise MyException(e, sys)

start_model_training()
