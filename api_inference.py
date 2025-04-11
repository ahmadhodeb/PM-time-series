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
    flight_phase_CRUISE: int
    flight_phase_TAKEOFF: int
    flight_phase_CLIMB: int

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse(content={"status": "healthy"})

@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        # Add preprocessing to match the scaler's expected input
        # Assuming 'flight_phase' was one-hot encoded during training
        flight_phase_encoded = [0, 0, 0]  # Adjusted to include 'CLIMB' encoding

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
            "flight_phase_TAKEOFF",
            "flight_phase_CLIMB"
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
                input_data.flight_phase_CRUISE,  # Use provided value for flight_phase_CRUISE
                input_data.flight_phase_TAKEOFF,  # Use provided value for flight_phase_TAKEOFF
                input_data.flight_phase_CLIMB   # Use provided value for flight_phase_CLIMB
            ]
        ])

        # Convert input data to a DataFrame with feature names
        input_df = pd.DataFrame(input_array, columns=feature_names)

        # Ensure input features match the model's expected features
        expected_features = [
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

        # Drop unexpected features and add missing ones with default values
        input_df = input_df.reindex(columns=expected_features, fill_value=0)

        # Scale the input data
        scaled_input = scaler.transform(input_df)

        # Make a prediction
        prediction = model.predict(scaled_input)

        # Convert prediction to a native Python float and round to an integer
        prediction_value = int(round(prediction[0]))

        # Log the prediction
        logging.info(f"Input: {input_data.model_dump()}, Prediction: {prediction_value}")

        return {"RUL_prediction": prediction_value}
    except ValueError as ve:
        logging.error(f"Validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

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