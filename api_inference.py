from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
from pydantic import BaseModel
from xgboost import XGBRegressor
import logging
from datetime import datetime
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import os

# Helper to load model and scaler with error handling
def load_model_and_scaler(model_path, scaler_path):
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise RuntimeError(f"Scaler file not found: {scaler_path}")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Load all models and scalers with robust checks
xgb_model, xgb_scaler = load_model_and_scaler("best_xgb_model.pkl", "scale_xgb.pkl")
rf_model, rf_scaler = load_model_and_scaler("best_rf_model.pkl", "scaler_rf.pkl")
etr_model, etr_scaler = load_model_and_scaler("best_etr_model.pkl", "scale_etr.pkl")

# Configure logging
logging.basicConfig(
    filename="predictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# Define the FastAPI app
app = FastAPI()

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust origins as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the input data model
class PredictionInput(BaseModel):
    flight_cycle: float
    egt_probe_average: float
    fuel_flw: float
    core_spd: float
    zpn12p: float
    vib_n1_1_bearing: float
    vib_n2_1_bearing: float
    vib_n2_turbine_frame: float
    flight_phase_CRUISE: int
    flight_phase_TAKEOFF: int
    flight_phase_CLIMB: int
    model_choice: str

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse(content={"status": "healthy"})

print('[DEBUG] API server started and api_inference.py loaded')

@app.post("/predict")
async def predict(input_data: PredictionInput):
    print('[DEBUG] /predict endpoint called')
    """
    Predict RUL using the selected model or the average of all three models.
    Model options: xgboost, randomforest, extratrees, average
    """
    try:
        print('[DEBUG] Entered try block in /predict')
        print("[DEBUG] Received input_data:", input_data)
        feature_names = [
            'flight_cycle',
            'egt_probe_average',
            'fuel_flw',
            'core_spd',
            'zpn12p',
            'vib_n1_#1_bearing',
            'vib_n2_#1_bearing',
            'vib_n2_turbine_frame',
            'flight_phase_CLIMB',
            'flight_phase_CRUISE',
            'flight_phase_TAKEOFF'
        ]
        # Build input dict and ensure all required features are present
        input_dict = {
            "flight_cycle": input_data.flight_cycle,
            "egt_probe_average": input_data.egt_probe_average,
            "fuel_flw": input_data.fuel_flw,
            "core_spd": input_data.core_spd,
            "zpn12p": input_data.zpn12p,
            "vib_n1_#1_bearing": input_data.vib_n1_1_bearing,
            "vib_n2_#1_bearing": input_data.vib_n2_1_bearing,
            "vib_n2_turbine_frame": input_data.vib_n2_turbine_frame,
            "flight_phase_CRUISE": getattr(input_data, "flight_phase_CRUISE", 0),
            "flight_phase_TAKEOFF": getattr(input_data, "flight_phase_TAKEOFF", 0),
            "flight_phase_CLIMB": getattr(input_data, "flight_phase_CLIMB", 0)
        }
        print("[DEBUG] Constructed input_dict:", input_dict)
        # Ensure all required columns are present (fill missing with 0)
        for col in feature_names:
            if col not in input_dict:
                print(f"[DEBUG] Missing column {col}, filling with 0")
                input_dict[col] = 0
        # Guarantee all dummy columns exist (for robustness)
        for col in ['flight_phase_CLIMB', 'flight_phase_CRUISE', 'flight_phase_TAKEOFF']:
            if col not in input_dict:
                print(f"[DEBUG] Missing dummy column {col}, filling with 0")
                input_dict[col] = 0
        # Ensure DataFrame uses the exact feature order
        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(columns=feature_names, fill_value=0)
        print("[DEBUG] Input DataFrame for prediction (ordered):")
        print(input_df)
        # All models expect the same features (order and names)
        X = input_df[feature_names]
        print("[DEBUG] Features to be scaled:")
        print(X)
        preds = []
        if input_data.model_choice == "xgboost":
            print("[DEBUG] Using XGBoost model and scaler")
            scaled = xgb_scaler.transform(X)
            print("[DEBUG] Scaled features:", scaled)
            pred = xgb_model.predict(scaled)
            print("[DEBUG] XGBoost prediction:", pred)
            preds.append(pred[0])
        elif input_data.model_choice == "randomforest":
            print("[DEBUG] Using Random Forest model and scaler")
            scaled = rf_scaler.transform(X)
            print("[DEBUG] Scaled features:", scaled)
            pred = rf_model.predict(scaled)
            print("[DEBUG] Random Forest prediction:", pred)
            preds.append(pred[0])
        elif input_data.model_choice == "extratrees":
            print("[DEBUG] Using Extra Trees model and scaler")
            scaled = etr_scaler.transform(X)
            print("[DEBUG] Scaled features:", scaled)
            pred = etr_model.predict(scaled)
            print("[DEBUG] Extra Trees prediction:", pred)
            preds.append(pred[0])
        elif input_data.model_choice == "average":
            print("[DEBUG] Using average of all models")
            scaled_xgb = xgb_scaler.transform(X)
            scaled_rf = rf_scaler.transform(X)
            scaled_etr = etr_scaler.transform(X)
            print("[DEBUG] Scaled features for XGBoost:", scaled_xgb)
            print("[DEBUG] Scaled features for Random Forest:", scaled_rf)
            print("[DEBUG] Scaled features for Extra Trees:", scaled_etr)
            pred_xgb = xgb_model.predict(scaled_xgb)[0]
            pred_rf = rf_model.predict(scaled_rf)[0]
            pred_etr = etr_model.predict(scaled_etr)[0]
            print(f"[DEBUG] XGBoost: {pred_xgb}, Random Forest: {pred_rf}, Extra Trees: {pred_etr}")
            preds.append(np.mean([pred_xgb, pred_rf, pred_etr]))
        else:
            print("[DEBUG] Invalid model_choice received:", input_data.model_choice)
            raise ValueError("Invalid model_choice. Choose from xgboost, randomforest, extratrees, average.")
        prediction_value = int(round(preds[0]))
        print("[DEBUG] Final prediction value:", prediction_value)
        logging.info(f"Input: {input_data.model_dump()}, Model: {input_data.model_choice}, Prediction: {prediction_value}")
        return {"RUL_prediction": prediction_value}
    except ValueError as ve:
        print('[DEBUG] ValueError in /predict:', ve)
        print("[DEBUG] ValueError:", ve)
        logging.error(f"Validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print('[DEBUG] Exception in /predict:', e)
        print("[DEBUG] Exception:", e)
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/start-server")
def start_server():
    try:
        # Add logic to initialize the server or perform setup tasks
        # For example, starting a background process or initializing resources
        return {"message": "Server started successfully!"}
    except Exception as e:
        logging.error(f"Error starting server: {e}")
        raise HTTPException(status_code=500, detail="Failed to start the server.")

@app.post("/initialize")
def initialize():
    try:
        # Logic to start the HTTP server and load the ONNX model
        # This is a placeholder for actual initialization logic
        return {"message": "Initialization successful!"}
    except Exception as e:
        logging.error(f"Error during initialization: {e}")
        raise HTTPException(status_code=500, detail="Initialization failed.")

@app.post("/initialize-all")
def initialize_all():
    try:
        # Perform all necessary initialization tasks
        # Example: Load ONNX model, start HTTP server, etc.
        return {"message": "All pre-work completed successfully!"}
    except Exception as e:
        logging.error(f"Error during full initialization: {e}")
        raise HTTPException(status_code=500, detail="Full initialization failed.")