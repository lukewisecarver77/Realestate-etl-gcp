"""
config.py — Centralized Configuration

Load all environment variables here using python-dotenv.
Other modules should import from config rather than reading os.environ directly.

Values to configure:
  GCP_PROJECT_ID       — your GCP project ID string
  BIGQUERY_DATASET     — name of the target BigQuery dataset
  BIGQUERY_TABLE       — name of the target BigQuery table
  WRITE_MODE           — "WRITE_TRUNCATE" or "WRITE_APPEND"
  DATA_SOURCE_URL      — direct URL to the Redfin TSV download
  RAW_DATA_PATH        — local path to save the raw file (data/raw/)
  PROCESSED_DATA_PATH  — local path to save cleaned file (data/processed/)
  FIGURES_PATH         — output path for saved charts (outputs/figures/)
  RANDOM_STATE         — integer seed for reproducibility across sklearn models
  N_CLUSTERS           — default K for K-Means (can be overridden by elbow method)
  TEST_SIZE            — train/test split ratio (e.g. 0.2 for 80/20)
"""
