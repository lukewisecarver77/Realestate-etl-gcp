import pandas as pd
from sklearn.preprocessing import LabelEncoder

def transform(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
  """Clean, normalize and engineer features from the raw Redfin DataFrame. Returns the cleaned df and a dict containing the label encoder mapping"""
  
  
  starting_rows = len(df)
  
  # Normalize column names
  df.columns = df.columns.str.lower().str.replace(" ", "_")
  
  # Drop unnecessary columns for analysis or ML
  cols_to_drop = [
        "region_type_id",
        "region_id",
        "adjusted_average_new_listings_yoy",
        "average_pending_sales_listing_updates_yoy",
        "off_market_in_two_weeks_yoy",
        "adjusted_average_homes_sold_yoy",
        "median_new_listing_price_yoy",
        "median_sale_price_yoy",
        "median_days_to_close_yoy",
        "median_new_listing_ppsf_yoy",
        "active_listings_yoy",
        "median_days_on_market_yoy",
        "percent_active_listings_with_price_drops_yoy",
        "age_of_inventory_yoy",
        "weeks_of_supply_yoy",
        "median_pending_sqft_yoy",
        "average_sale_to_list_ratio_yoy",
        "median_sale_ppsf_yoy",
        "last_updated",
    ]
  df = df.drop(columns=cols_to_drop)
  
  # Parse the date columns
  df["period_begin"] = pd.to_datetime(df["period_begin"])
  df["period_end"] = pd.to_datetime(df["period_end"])
  
  # Drop rows where critical columns are null
  critical_cols = ["median_sale_price", "median_sale_ppsf", "region_name"]
  rows_before = len(df)
  df = df.dropna(subset=critical_cols)
  rows_after = len(df)
  print(f"Dropped {rows_before - rows_after:,} rows with nulls in critical columns")
  
  # Filter to national and metro level only 
  rows_before = len(df)
  df = df[df["region_type"].isin(["national", "metro"])]
  df = df[df["region_name"] != "All Redfin Metros"]
  rows_after = len(df)
  print(f"Dropped {rows_before - rows_after:,} rows outside national/metro region type")
  
  
  # Engineer derived features
  
  # Remove price outliers
  df = df[df["median_sale_price"] < 2000000]
  df = df[df["median_sale_ppsf"] < 2000]
  
  # Month for seaonality
  df["month"] = df["period_end"].dt.month
  
  # Price per square foot, derive my own for transparency
  df["price_per_sqft"] = df["median_sale_price"] / df["median_pending_sqft"]
  
  
  # Days on market bucket
  df["dom_bucket"] = pd.cut(
    df["median_days_on_market"],
    bins=[0, 15, 45, float("inf")],
    labels=["fast (<15d)", "normal (15-45d)", "slow (>45d)"],
  )
  
  # Price tier (low / mid / high based on percentiles)
  df["price_tier"] = pd.qcut(
    df["median_sale_price"],
    q=3,
    labels=["low", "mid", "high"]
  )
  
  # Keep human readable region name before encoding
  df["region_name_label"] = df["region_name"]
  
  # Encode region_name for ML
  le = LabelEncoder()
  df["region_name_encoded"] = le.fit_transform(df["region_name"])
  
  # Drop rows with nulls in ML feature columns
  df = df.dropna(subset=["median_days_on_market", "median_pending_sqft", "price_per_sqft", "dom_bucket"])
  print(f"After dropping feature nulls: {len(df):,} rows")
  
  # Store mapping to decode later
  encoder_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
  
  # Final row count log
  print(f"Transform complete: {starting_rows:,} -> {len(df):,} rows")
  print(f"Final shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
  
  return df, encoder_mapping



# Standalone testing

if __name__ == "__main__":
    from pipeline.extract import extract
 
    raw_path = "data/raw"
    filename = "data/raw/weekly_housing_market_data_most_recent.tsv000"
 
    df_raw = pd.read_csv(filename, sep="\t")
    df_clean, mapping = transform(df_raw)
 
    print(df_clean.head())
    print(f"\nSample encoder mapping (first 5): {dict(list(mapping.items())[:5])}")