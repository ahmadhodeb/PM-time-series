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

# Load the trained model and scaler
model = joblib.load("best_xgb_model.pkl")
scaler = joblib.load("scale_xgb.pkl")

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

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse(content={"status": "healthy"})

@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        # Add preprocessing to match the scaler's expected input
        # Assuming 'flight_phase' was one-hot encoded during training
        flight_phase_encoded = [0, 0]  # Example: Adjust based on actual encoding

        # Correct feature names to match training data
        feature_names = [
            "flight_cycle",
            "egt_probe_average",
            "fuel_flw",
            "core_spd",
            "zpn12p",
            "vib_n1_#1_bearing",
            "vib_n2_#1_bearing",
            "vib_n2_turbine_frame",
            "flight_phase_CRUISE",
            "flight_phase_TAKEOFF"
        ]

        # Update input array to include correct feature names
        input_array = np.array([
            [
                input_data.flight_cycle,
                input_data.egt_probe_average,
                input_data.fuel_flw,
                input_data.core_spd,
                input_data.zpn12p,
                input_data.vib_n1_1_bearing,  # Map to vib_n1_#1_bearing
                input_data.vib_n2_1_bearing,  # Map to vib_n2_#1_bearing
                input_data.vib_n2_turbine_frame,
                0,  # Placeholder for flight_phase_CRUISE
                0   # Placeholder for flight_phase_TAKEOFF
            ]
        ])

        # Convert input data to a DataFrame with feature names
        input_df = pd.DataFrame(input_array, columns=feature_names)

        # Scale the input data
        scaled_input = scaler.transform(input_df)

        # Make a prediction
        prediction = model.predict(scaled_input)

        # Convert prediction to a native Python float
        prediction_value = float(prediction[0])

        # Log the prediction
        logging.info(f"Input: {input_data.model_dump()}, Prediction: {prediction_value}")

        return {"RUL_prediction": prediction_value}
    except ValueError as ve:
        logging.error(f"Validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")