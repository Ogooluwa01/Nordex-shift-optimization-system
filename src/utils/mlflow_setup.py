import dagshub
import mlflow
import os

from dotenv import load_dotenv

def setup_mlflow_connection():
  dagshub.init(repo_owner='aleyiboogooluwamary',
                  repo_name='Nordex-shift-optimization-system',
                    mlflow=True)
  mlflow.set_experiment("Nordex_shift_optimization_Production_Models")

