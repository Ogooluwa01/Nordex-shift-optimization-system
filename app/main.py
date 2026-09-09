import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import threading
import optuna
import datetime

from src.logger import configure_logger
from src.pipelines.training_pipeline import start_model_training
from src.exception import MyException
from src.utils.schema_loader import read_yaml
from src.pipelines.prediction_pipeline import prediction_pipeline
from utils.model_utils import load_model_from_mlflow

logging = configure_logger()

app = FastAPI(title="NordexShift Optimization API")

model = None  # Global variable to hold the trained model

@app.on_event("startup")
def load_model_on_startup():
    global model
    try:
        logging.info("loading model at startup...")
        model = load_model_from_mlflow()
        logging.info("Model loaded successfully at startup.")
    except Exception as e:
        logging.error(f"Error loading model at startup: {e}")
        raise MyException(e, sys)

class ShiftInput(BaseModel):
    units_produced: int
    defect_count: int
    cycle_time_avg: float
    experience_level: int
    runtime_hours: int
    downtime_minutes: int
    maintenance_downtime: int
    temperature: float
    humidity: float
    shift_duration: float
    downtime_ratio: float
    defect_rate: float
    shift_name: str
    skill_category: str
    machine_status: str
    issue_type: str
    defect_type: str
    severity: str
    inspection_result: str

class OptimizationInput(BaseModel):
    shift_name: str
    skill_category: str
    machine_status: str
    issue_type = str
    defect_type = str
    severity = str
    inspection_result = str

    experience_range:tuple[int, int]
    downtime_range:tuple[float, float]
    defect_range:tuple[int, int]
    n_trials: int 

@app.get("/")
def health_check():
    return {"message": "Nordex shift optimization API is running"}

# prediction
@app.post("/predict")
def predict_shift_efficiency_score(input: ShiftInput):
    try:
        if model is None:
            raise Exception("Model is not loaded")
        result = prediction_pipeline(input.dict(), model)

        return {
            "predicted_shift_efficiency_score": float(result[0])
        }
    except Exception as e:
        logging.error(f"prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# optimization section
@app.post("/optimize")
def optimize_shift_(input: OptimizationInput):
    try:
        if model is None:
            raise Exception("Model is not loaded")
        def objective(trial):
           experience_level = trial.suggest_int(
               "experience_level",
                 input.experience_range[0], 
                 input.experience_range[1]
                 )
           downtime_minutes = trial.suggest_int(
                "downtime_minutes",
                  input.downtime_range[0], 
                  input.downtime_range[1]
                  )
           defect_count = trial.suggest_int(
                "defect_count",
                  input.defect_range[0], 
                  input.defect_range[1]
                  )
           units_produced = trial.suggest_int(
                "units_produced",
                  600, 
                  1200
                  )
           maintenance_downtime = trial.suggest_int(
               "maintenance_downtime", 0, 60)
           cycle_time_avg = trial.suggest_float(
                "cycle_time_avg", 30.0, 45.0)
           temperature = trial.suggest_float(
                "temperature", 18.0, 30.0)
           humidity = trial.suggest_float(
                "humidity", 30.0, 70.0)
           runtime_hours = 6.0
           total_machine_hours = runtime_hours
           shift_duration = runtime_hours
           day_of_week = datetime.datetime.today().weekday()
           defect_rate = trial.suggest_float(
               "defect_rate", 0, 2
           )
           hour_of_day = trial.suggest_float(
               "hour_of_day", 0, 24
           )
           skill_category = input.skill_category
           machine_status = input.machine_status
           issue_type = input.issue_type
           defect_type = input.defect_type
           severity = input.severity
           inspection_result = input.inspection_result


           data = {
               "experience_level": experience_level,
               "downtime_minutes": downtime_minutes,
               "defect_count": defect_count,
               "unit_produced": units_produced,
               "maintenance_downtime":maintenance_downtime,
               "cycle_time_avg": cycle_time_avg,
               "temperature": temperature,
               "humidity": humidity,
               "total_machine_hours": total_machine_hours,
               "shift_duration": shift_duration,
               "day_of_week": day_of_week,
               "defect_rate": defect_rate,
               "hour_of_day": hour_of_day,
               "skill_category": skill_category,
               "machine_status": machine_status,
               "issue_type": issue_type,
               "defect_type": defect_type,
               "severity": severity
               "inspection_result": inspection_result
           }

           pred = prediction_pipeline(data, model)

           return pred[0]
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=input.n_trial)

        return{
            "best_parameter": study.best_params,
            "best_score": study.best_value,
            "top_trails": study.trials_dataframe().sort_values("value", ascending=False)
            .to_dict(orient="records")
        }
    except Exception as e:
        logging.error(f"error occured during optimization process.. ")
        raise HTTPException(status_code=500, detail=str(e))

def retrain_pipeline():
    global model
    try:
        start_model_training()
        logging.info("reloading model after training....")
        model = load_model_from_mlflow()

    except Exception as e:
        logging.error(f"Traning failed {e}")

@app.post("/retrain")
def retrain_model():
    thread = threading.Thread(target=retrain_pipeline)
    thread.start()

    return {
        "message": "Threading started in background, model will be updated automatically."
    }