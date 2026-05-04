import os
import requests
import pandas as pd
from datetime import datetime


def extract(url: str, raw_data_path: str) -> pd.DataFrame:
  """ Download Redfin TSV file, save locally, read into dataframe, and return it """
  
  # Download file
  print(f"Downloading data from {url}..")
  response = requests.get(url, stream=True)
  
  if response.status_code != 200:
    raise Exception(f"Download failed with status code {response.status_code}")
  
  # timestamp for file and save to data/raw
  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  filename = os.path.join(raw_data_path, f"metro_market_{timestamp}.tsv.gz")
  
  os.makedirs(raw_data_path, exist_ok=True)
  
  with open(filename, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
      f.write(chunk)
      
  print(f"Saved raw file to: {filename}")
  
  
  # Read into dataframe
  df = pd.read_csv(filename, sep="\t")
  
  # Log shape and file size
  file_size_mb = os.path.getsize(filename) / (1024 * 1024)
  print(f"Loaded DataFrame: {df.shape[0]:,} rows, {df.shape[1]} columns")
  print(f"File size on disk: {file_size_mb:.1f} MB")
  
  return df

# Standalone testing

# if __name__ == "__main__":
#     from dotenv import load_dotenv
#     load_dotenv()
 
#     url = os.getenv("DATA_SOURCE_URL")
#     raw_path = "data/raw"
 
#     df = extract(url, raw_path)
#     print(df.head())


if __name__ == "__main__":
    raw_path = "data/raw"
    filename = "data/raw/weekly_housing_market_data_most_recent.tsv000"
    df = pd.read_csv(filename, sep="\t")
    print(f"{df.shape[0]:,} rows, {df.shape[1]} columns")
    print(df.head())