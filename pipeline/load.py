import os
import pandas as pd
from google.cloud import bigquery


def load(df: pd.DataFrame, project_id: str, dataset_id: str, table_id: str, write_mode: str = "WRITE_TRUNCATE") -> None:
  """Load a cleaned DataFrame into a BigQuery Talbe. Creates the dataset if it doesn't already exist"""
  
  client = bigquery.Client(project=project_id)
  
  # Create dataset if it doesn't exist
  dataset_ref = f"{project_id}.{dataset_id}"
  dataset = bigquery.Dataset(dataset_ref)
  dataset.location = "US"
  
  client.create_dataset(dataset, exists_ok=True)
  print(f"Dataset ready: {dataset_ref}")
  
  # Define full table for reference
  table_ref = f"{project_id}.{dataset_id}.{table_id}"
  
  # Configure the load job
  job_config = bigquery.LoadJobConfig(
    write_disposition=write_mode,
    autodetect=True,
  )
  
  # Convert categoricals to string because BigQuery doesn't accept pandas categoricals
  for col in df.select_dtypes(include="category").columns:
    df[col] = df[col].astype(str)
  
  # Load DataFrame into BigQuery
  print(f"Loading {len(df):,} rows into {table_ref}")
  job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
  job.result()
  
  # Confirm row count
  table = client.get_table(table_ref)
  print(f"Load complete: {table.num_rows:,} rows now in {table_ref}")
  
# --- Standalone testing ---
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
 
    from pipeline.extract import extract
    from pipeline.transform import transform
    from pipeline.validate import validate
 
    filename = "data/raw/weekly_housing_market_data_most_recent.tsv000"
    df_raw = pd.read_csv(filename, sep="\t")
    df_clean, mapping = transform(df_raw)
    validate(df_clean)
 
    load(
        df=df_clean,
        project_id=os.getenv("GCP_PROJECT_ID"),
        dataset_id=os.getenv("BIGQUERY_DATASET"),
        table_id=os.getenv("BIGQUERY_TABLE"),
        write_mode=os.getenv("WRITE_MODE", "WRITE_TRUNCATE"),
    )