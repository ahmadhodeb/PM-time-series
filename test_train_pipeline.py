import pytest
import subprocess

def test_train_pipeline():
    # Add test logic for the training pipeline
    try:
        subprocess.run(['python', 'train_pipeline.py'], check=True)
    except subprocess.CalledProcessError:
        pytest.fail("Training pipeline failed")