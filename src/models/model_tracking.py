import dagshub
import mlflow.tracking
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from src.logger import configure_logger
from src.exception import MyException
import os
import sys
from src.utils.mlflow_setup import setup_mlflow_connection
import mlflow
from src.utils.model_utils import get_existing_model_metrics


logging = configure_logger()

class ModelTracker:
    def __init__(self):
        setup_mlflow_connection()

        self.client = MlflowClient()
        self.registered_model_name = "NordexShiftOptimizationModel"

    def push_model(self, model, r2_score: float, mae_score: float) -> bool:
            """
            pushing model to mlflow registry and promote to production.
            """
            try:
                logging.info("Checking existing model performance in MLFLOW")
                existing_model_mae, existing_model_r2 = get_existing_model_metrics(
                    self.registered_model_name
                )

                push_new_model = False

                # model performance comparison
                if existing_model_r2 is None:
                    push_new_model = True
                    logging.info("No existing model found. Proceeding to registerthe new model.")
                elif r2_score > existing_model_r2 and mae_score < existing_model_mae:                 
                     push_new_model = True
                     logging.info("New model outperforms the existing model. Proceeding to register the new model.")
                elif r2_score == existing_model_r2 and mae_score < existing_model_mae:
                    push_new_model = True
                    logging.info("Both model has the equal R2_score but lower MAE than the existing model. Proceeding to register the new model.")
                else:
                    logging.info("Existing model outperforms the new model. Skipping registration of the new model.")
                    return False
                with mlflow.start_run():
                    logging.info("Logging the new model to MLflow...")
                    mlflow.log_metric("r2_score", r2_score)
                    mlflow.log_metric("mae_score", mae_score)

                    mlflow.sklearn.log_model(
                         sk_model=model,
                         artifact_path="model",
                         registered_model_name=self.registered_model_name

                    )
                    logging.info("Model Logged Sucessfully")

                    return True
            except Exception as e:
                logging.error(f"Error occurred while pushing the model: {str(e)}")
                raise MyException(e, sys)

            

