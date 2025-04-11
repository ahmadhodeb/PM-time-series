import os
import pandas as pd
import json
from evidently import Report
from evidently.presets import DataDriftPreset

# Define a function to check for data drift
def check_data_drift(reference_data, current_data):
    # Create a data drift report
    report = Report([DataDriftPreset()])
    report.run(current_data=current_data, reference_data=reference_data)

    # Generate the report and extract it as JSON
    report_data = report.as_dict()

    # Check if data drift is detected
    drift_detected = report_data["metrics"]["data_drift"]["data_drift_detected"]
    return drift_detected