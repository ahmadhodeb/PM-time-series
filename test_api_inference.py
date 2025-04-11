import pytest
from fastapi.testclient import TestClient
from api_inference import app

# Initialize the TestClient
client = TestClient(app)

def test_health_check():
    """Test the /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_valid_input():
    """Test the /predict endpoint with valid input."""
    valid_input = {
        "flight_cycle": 100,
        "egt_probe_average": 500,
        "fuel_flw": 200,
        "core_spd": 3000,
        "zpn12p": 1.5,
        "vib_n1_1_bearing": 0.02,
        "vib_n2_1_bearing": 0.03,
        "vib_n2_turbine_frame": 0.04
    }
    response = client.post("/predict", json=valid_input)
    assert response.status_code == 200
    assert "RUL_prediction" in response.json()

def test_predict_invalid_input():
    """Test the /predict endpoint with invalid input."""
    invalid_input = {
        "flight_cycle": "invalid",
        "egt_probe_average": 500,
        "fuel_flw": 200,
        "core_spd": 3000,
        "zpn12p": 1.5,
        "vib_n1_1_bearing": 0.02,
        "vib_n2_1_bearing": 0.03,
        "vib_n2_turbine_frame": 0.04
    }
    response = client.post("/predict", json=invalid_input)
    assert response.status_code == 422

def test_api_inference():
    # Add test logic for API inference
    assert True  # Replace with actual assertions