# Predictive Maintenance for Aircraft Engines

## Overview
This project focuses on predictive maintenance for aircraft engines using machine learning and deep learning techniques. The goal is to predict the Remaining Useful Life (RUL) of engines based on sensor data, enabling proactive maintenance and reducing downtime.

## Features
- **Data Cleaning and Preprocessing**: Includes scripts for cleaning and preparing raw engine data.
- **Exploratory Data Analysis (EDA)**: Visualizations and insights into the dataset.
- **Machine Learning Models**: Implementation of XGBoost for RUL prediction.
- **Deep Learning Models**: Neural network-based approaches for RUL prediction.
- **Model Deployment**: A FastAPI-based backend for serving predictions.
- **Monitoring**: Integration with Prometheus for monitoring data drift and model performance.
- **Prediction Tool**: A user-friendly web interface for predicting the Remaining Useful Life (RUL) of engines.

## Project Structure
```
├── api_inference.py          # FastAPI backend for serving predictions
├── best_xgb_model.onnx       # ONNX model for inference
├── best_xgb_model.pkl        # Trained XGBoost model
├── data_drift.py             # Script for monitoring data drift
├── Dockerfile                # Docker configuration for containerization
├── index.html                # Frontend for user interaction
├── prediction tool.url       # Shortcut to access the prediction tool
├── predictions.log           # Log file for predictions
├── requirements.txt          # Python dependencies
├── start_servers.py          # Script to start backend and HTTP server
├── train_pipeline.py         # Training pipeline for machine learning models
├── test_*.py                 # Unit tests for various components
├── Data/                     # Folder containing raw and cleaned datasets
├── mlruns/                   # MLflow tracking directory
└── notebooks/                # Jupyter notebooks for EDA and model development
```

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd PM-time-series
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the Backend and HTTP Server**:
   ```bash
   python start_servers.py
   ```

4. **Access the Prediction Tool**:
   - Double-click the `prediction tool.url` shortcut to open the prediction tool in your default browser.
   - Alternatively, navigate to `http://localhost:8000/index.html` in your browser.

## Running the Localhost Server

To run the localhost server and access the prediction tool, follow these steps:

1. **Ensure All Dependencies Are Installed**:
   - Make sure you have Python installed on your system (version 3.8 or higher).
   - Install the required Python packages by running:
     ```bash
     pip install -r requirements.txt
     ```

2. **Start the Backend and HTTP Server**:
   - Open a terminal or command prompt.
   - Navigate to the project directory:
     ```bash
     cd "..\PM-time-series"
     ```
   - Run the following command to start the backend and HTTP server:
     ```bash
     python start_servers.py
     ```
   - This script will:
     - Free up port 8000 if it is already in use.
     - Start the FastAPI backend server for predictions.
     - Start an HTTP server to serve the `index.html` file and other resources.

3. **Access the Prediction Tool**:
   - Once the server is running, open the prediction tool in your browser:
     - Double-click the `prediction tool.url` shortcut in the project directory.
     - Or manually navigate to `http://localhost:8000/index.html` in your browser.

4. **Using the Prediction Tool**:
   - Enter the required engine parameters in the web interface.
   - Click the **Predict** button to get the Remaining Useful Life (RUL) prediction.

5. **Stopping the Server**:
   - To stop the server, press `Ctrl + C` in the terminal where the `start_servers.py` script is running.

## Using the Prediction Tool
1. Open the prediction tool using the steps above.
2. Enter the required engine parameters in the web interface:
   - **Flight Cycle**: Number of flight cycles completed.
   - **EGT Probe Average**: Average Exhaust Gas Temperature.
   - **Fuel Flow**: Fuel flow rate.
   - **Core Speed**: Core engine speed.
   - **ZPN12P**: A specific engine parameter.
   - **Vibration N1 #1 Bearing**: Vibration level of N1 #1 bearing.
   - **Vibration N2 #1 Bearing**: Vibration level of N2 #1 bearing.
   - **Vibration N2 Turbine Frame**: Vibration level of N2 turbine frame.
   - **Flight Phase**: Select the current flight phase (CRUISE, TAKEOFF, or CLIMB).
3. Click the **Predict** button.
4. View the predicted Remaining Useful Life (RUL) and health indicator in the results section.

## Key Files
- `api_inference.py`: Backend for serving predictions.
- `index.html`: Frontend for user interaction.
- `start_servers.py`: Script to start all necessary services.
- `best_xgb_model.onnx`: ONNX model for inference.
- `train_pipeline.py`: Training pipeline for machine learning models.

## Testing
Run the test suite to ensure all components are functioning correctly:
```bash
pytest
```

## Future Work
- Enhance the deep learning models for better accuracy.
- Implement real-time data ingestion and processing.
- Extend monitoring capabilities with advanced metrics.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- The dataset used in this project is sourced from publicly available engine data.
- Special thanks to the contributors and the open-source community for their support.
