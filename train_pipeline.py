import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
import mlflow.sklearn
import xgboost as xgb
import os

# Load the data
data_path = 'Data/engines2_data_cleaned_no_outliers.csv'
df = pd.read_csv(data_path)

# Feature selection and preprocessing
features = ['flight_cycle', 'egt_probe_average', 'fuel_flw', 'core_spd', 'zpn12p', 'vib_n1_#1_bearing', 'vib_n2_#1_bearing', 'vib_n2_turbine_frame']
target = 'RUL'

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Set MLflow tracking URI dynamically
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

# Set experiment name dynamically
experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "Predictive_Maintenance")
mlflow.set_experiment(experiment_name)

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("data_path", data_path)
    mlflow.log_param("features", features)
    mlflow.log_param("target", target)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }

    xgb_model = xgb.XGBRegressor(random_state=42)
    grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2)
    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_

    # Log hyperparameters
    for param, values in param_grid.items():
        mlflow.log_param(param, values)

    # Log the best model's hyperparameters
    best_params = grid_search.best_params_
    for param, value in best_params.items():
        mlflow.log_param(f"best_{param}", value)

    # Verify the type of the scaler before saving
    print(f"Type of scaler before saving: {type(scaler)}")

    # Log the scaler
    import joblib
    joblib.dump(scaler, "scale_xgb.pkl")
    mlflow.log_artifact("scale_xgb.pkl")

    # Predictions
    y_pred = best_model.predict(X_test_scaled)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    # Log metrics
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)

    # Log the model
    mlflow.sklearn.log_model(best_model, "model")

    print(f"Model trained. MAE: {mae}, RMSE: {rmse}")