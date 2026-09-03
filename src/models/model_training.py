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
from src.models.model_tracking import ModelTracker
from src.logger import configure_logger
from src.exception import MyException
from src.utils.schema_loader import read_yaml
from sklearn.ensemble import GradientBoostingRegressor
from config.constant import SCHEMA_PATH

logging = configure_logger()

class ModelTrainer:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.schema = read_yaml(SCHEMA_PATH)
        self.model_pipeline = None

    def build_training_pipeline(self):
        """ building a full pipeline including the preprocessing method and mosel """
        numerical_Columns = self.schema['columns']['numerical_columns']
        categorical_Columns = self.schema['columns']['categorical_columns']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numerical_Columns),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_Columns)
            ]
        )
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', GradientBoostingRegressor())
        ])
        return pipeline

    def train_model(self):
        try:
            logging.info("Starting model training pipeline...")
            self.model_pipeline = self.build_training_pipeline()
            self.model_pipeline.fit(self.X_train, self.y_train)
            logging.info("Model training completed successfully.")
            return self.model_pipeline
        except Exception as e:
            logging.error(f"Error occurred during model training: {e}")
            raise MyException(e, sys)

    def evaluate_model(self):
        try:
            logging.info("Evaluating the trained pipeline...")
            y_pred = self.model_pipeline.predict(self.X_test)
            r2 = sklearn.metrics.r2_score(self.y_test, y_pred)
            mae = sklearn.metrics.mean_absolute_error(self.y_test, y_pred)
            logging.info(f"Pipeline evaluation completed with metrics of R2: {r2}, MAE: {mae}")
            return r2, mae
        except Exception as e:
            logging.error(f"Error occurred during model evaluation: {str(e)}")
            raise MyException(e, sys)

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

    