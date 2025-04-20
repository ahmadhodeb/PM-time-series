from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import subprocess
from data_drift import check_data_drift

def retrain_model():
    # Call the updated pipeline that trains all models
    subprocess.run(['python', 'train_pipeline.py'], check=True)

def register_model():
    # Logic to register the retrained model in MLflow
    pass

def notify():
    # Logic to send notification about retraining status
    pass

def check_drift_and_retrain():
    if check_data_drift():
        retrain_model()
    else:
        print("No data drift detected. Retraining not required.")

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'retrain_model_dag',
    default_args=default_args,
    description='A DAG to retrain the predictive maintenance model',
    schedule_interval=timedelta(days=7),
    start_date=datetime(2025, 4, 11),
    catchup=False,
)

check_drift_task = PythonOperator(
    task_id='check_drift_and_retrain',
    python_callable=check_drift_and_retrain,
    dag=dag,
)

retrain_task = PythonOperator(
    task_id='retrain_model',
    python_callable=retrain_model,
    dag=dag,
)

register_task = PythonOperator(
    task_id='register_model',
    python_callable=register_model,
    dag=dag,
)

notify_task = PythonOperator(
    task_id='notify',
    python_callable=notify,
    dag=dag,
)

check_drift_task >> retrain_task >> register_task >> notify_task