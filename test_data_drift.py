import pytest
from data_drift import check_data_drift
import pandas as pd
import numpy as np

def test_check_data_drift():
    # Load datasets
    reference_data = pd.read_csv("Data/engines2_data_cleaned_no_outliers.csv")
    current_data = pd.read_csv("Data/engines_data_cleaned.csv")

    # Test the data drift detection function
    drift_detected = check_data_drift(reference_data, current_data)
    assert isinstance(drift_detected, bool), "Drift detection should return a boolean value"
    # Add more assertions based on expected behavior

def test_check_data_drift_with_mock_data():
    # Create mock data to simulate data drift
    reference_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(5, 2, 1000)
    })

    current_data = pd.DataFrame({
        'feature1': np.random.normal(1, 1, 1000),  # Simulate drift in feature1
        'feature2': np.random.normal(5, 2, 1000)
    })

    # Call the function with mock data
    drift_detected = check_data_drift(reference_data, current_data)

    # Assert that drift is detected
    assert drift_detected is True, "Drift should be detected for feature1"

def test_check_data_drift_no_drift():
    # Create mock data with no drift
    reference_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(5, 2, 1000)
    })

    current_data = reference_data.copy()  # No drift

    # Call the function with mock data
    drift_detected = check_data_drift(reference_data, current_data)

    # Assert that no drift is detected
    assert drift_detected is False, "No drift should be detected when data is identical"

def test_check_data_drift_invalid_input():
    # Test with invalid input
    with pytest.raises(ValueError):
        check_data_drift(None, None)