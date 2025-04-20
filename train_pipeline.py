import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
import mlflow.sklearn
import xgboost as xgb
import os
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import joblib
import pickle
# Load the data
data_path = 'Data/engines2_data_cleaned_no_outliers.csv'
df = pd.read_csv(data_path)

# Feature selection and preprocessing
features = ['flight_cycle', 'egt_probe_average', 'fuel_flw', 'core_spd', 'zpn12p',
            'vib_n1_#1_bearing', 'vib_n2_#1_bearing', 'vib_n2_turbine_frame',
            'flight_phase_CLIMB', 'flight_phase_CRUISE', 'flight_phase_TAKEOFF']
target = 'RUL'

# One-hot encode the 'flight_phase' column
X = pd.get_dummies(df, columns=['flight_phase'], drop_first=False)

X = X[features]
y = df[target]

# Data scaling using MinMaxScaler as in the notebook
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 80/20 split, then 50/50 split for val/test as in notebook
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

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

    # --- XGBoost ---
    xgb_param_grid = {
        'n_estimators': [300],
        'reg_alpha': [0],
        'reg_lambda': [10]
    }
    xgb_model = xgb.XGBRegressor(random_state=42)
    xgb_grid = GridSearchCV(xgb_model, xgb_param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2)
    xgb_grid.fit(X_train, y_train)
    best_xgb = xgb_grid.best_estimator_
    joblib.dump(best_xgb, 'best_xgb_model.pkl', protocol=pickle.HIGHEST_PROTOCOL, compress=4)
    joblib.dump(scaler, 'scale_xgb.pkl')
    y_pred_xgb = best_xgb.predict(X_test)
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    rmse_xgb = mean_squared_error(y_test, y_pred_xgb, squared=False)
    mlflow.log_metric("xgb_mae", mae_xgb)
    mlflow.log_metric("xgb_rmse", rmse_xgb)

    # --- Random Forest ---
    rf_param_grid = {
        'n_estimators': [200],
        'max_depth': [None],
        'min_samples_split': [2],
        'min_samples_leaf': [20],
    }
    rf = RandomForestRegressor(random_state=42)
    rf_grid = GridSearchCV(rf, rf_param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2)
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    joblib.dump(best_rf, 'best_rf_model.pkl', protocol=pickle.HIGHEST_PROTOCOL, compress=4)
    joblib.dump(scaler, 'scaler_rf.pkl')
    y_pred_rf = best_rf.predict(X_test)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
    mlflow.log_metric("rf_mae", mae_rf)
    mlflow.log_metric("rf_rmse", rmse_rf)

    # --- Extra Trees ---
    etr_param_grid = {
        'n_estimators': [200],
        'max_depth': [None],
        'min_samples_split': [2],
        'min_samples_leaf': [20],
    }
    etr = ExtraTreesRegressor(random_state=42)
    etr_grid = GridSearchCV(etr, etr_param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2)
    etr_grid.fit(X_train, y_train)
    best_etr = etr_grid.best_estimator_
    joblib.dump(best_etr, 'best_etr_model.pkl', protocol=pickle.HIGHEST_PROTOCOL, compress=4)
    joblib.dump(scaler, 'scale_etr.pkl')
    y_pred_etr = best_etr.predict(X_test)
    mae_etr = mean_absolute_error(y_test, y_pred_etr)
    rmse_etr = mean_squared_error(y_test, y_pred_etr, squared=False)
    mlflow.log_metric("etr_mae", mae_etr)
    mlflow.log_metric("etr_rmse", rmse_etr)

    print(f"XGBoost: MAE={mae_xgb}, RMSE={rmse_xgb}")
    print(f"Random Forest: MAE={mae_rf}, RMSE={rmse_rf}")
    print(f"Extra Trees: MAE={mae_etr}, RMSE={rmse_etr}")