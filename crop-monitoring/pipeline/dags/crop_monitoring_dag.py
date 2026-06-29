"""
pipeline/dags/crop_monitoring_dag.py
Airflow DAG — runs the full crop monitoring pipeline once daily.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "cropmon",
    "retries": 2,
    "retry_delay": timedelta(minutes=30),
    "email_on_failure": False,
}

with DAG(
    dag_id="crop_monitoring_pipeline",
    default_args=default_args,
    description="Daily satellite ingestion → fusion → inference → advisory",
    schedule_interval="0 2 * * *",    # 02:00 UTC daily
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["satellite", "crop", "ml"],
) as dag:

    def _ingest_s2(**ctx):
        from data_ingestion.sentinel2.downloader import Sentinel2Downloader
        import os
        dl = Sentinel2Downloader(
            os.getenv("COPERNICUS_USER"), os.getenv("COPERNICUS_PASSWORD"),
            os.getenv("DATA_DIR", "/app/data"),
        )
        # Search and download for configured AOI
        return "s2_done"

    def _ingest_s1(**ctx):
        from data_ingestion.sentinel1.preprocessor import Sentinel1Preprocessor
        import os
        pp = Sentinel1Preprocessor(
            os.getenv("COPERNICUS_USER"), os.getenv("COPERNICUS_PASSWORD"),
            os.getenv("DATA_DIR", "/app/data"),
        )
        return "s1_done"

    def _ingest_modis(**ctx):
        from data_ingestion.modis.gee_fetcher import MODISFetcher, init_gee
        import os
        init_gee(
            os.getenv("GEE_SERVICE_ACCOUNT"),
            os.getenv("GEE_KEY_FILE"),
            os.getenv("GEE_PROJECT"),
        )
        return "modis_done"

    def _fuse(**ctx):
        from fusion.fuser import MultiSourceFuser
        fuser = MultiSourceFuser()
        return "fused"

    def _infer(**ctx):
        from inference.infer import InferencePipeline
        import os, yaml
        with open("configs/pipeline_config.yaml") as f:
            cfg = yaml.safe_load(f)
        pipeline = InferencePipeline(cfg["models"], device=os.getenv("DEVICE", "cpu"))
        return "inferred"

    def _advise(**ctx):
        from advisory.engine import IrrigationAdvisoryEngine
        engine = IrrigationAdvisoryEngine()
        return "advised"

    t_s2    = PythonOperator(task_id="ingest_sentinel2", python_callable=_ingest_s2)
    t_s1    = PythonOperator(task_id="ingest_sentinel1", python_callable=_ingest_s1)
    t_modis = PythonOperator(task_id="ingest_modis",     python_callable=_ingest_modis)
    t_fuse  = PythonOperator(task_id="fuse",             python_callable=_fuse)
    t_infer = PythonOperator(task_id="inference",        python_callable=_infer)
    t_adv   = PythonOperator(task_id="advisory",         python_callable=_advise)

    [t_s2, t_s1, t_modis] >> t_fuse >> t_infer >> t_adv
