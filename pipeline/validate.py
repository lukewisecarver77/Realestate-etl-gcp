import pandas as pd

REQUIRED_COLUMNS = [
    "period_begin",
    "period_end",
    "region_type",
    "region_name",
    "median_sale_price",
    "median_sale_ppsf",
    "median_days_on_market",
    "active_listings",
    "median_pending_sqft",
    "price_per_sqft",
    "dom_bucket",
    "price_tier",
    "month",
    "region_name_label",
    "region_name_encoded",
]

def validate(df: pd.DataFrame) -> bool:
  """Run quality checks on the cleaned DataFrame before loading BigQuery
      logs pass or fail for each check and raises ValueError if any check fails"""
      
  print("Running validation Checks")
  passed = 0
  failed = 0
  
  # Schema check
  missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
  if missing_cols:
    failed += 1
    print(f" Failed schema check: missing columns: {missing_cols}")
    raise ValueError(f"Schema check failed. Missing columns: {missing_cols}")
  else:
    passed += 1
    print(f"Passed the Schemce Check: all columns present")
    
    
  # Null check on required columns
  nulls = df[REQUIRED_COLUMNS].isnull().sum()
  cols_with_nulls = nulls[nulls > 0]
  if not cols_with_nulls.empty:
    failed += 1
    print(f"Failed null check: nuls found in: {cols_with_nulls.to_dict()}")
    raise ValueError(f"Null check failed. Nulls found in required colums: {cols_with_nulls}")
  else:
    passed += 1
    print(f"Passed Null checks: no nulls in required columns")
  
  
  # Range Check
  invalid_price = df[df["median_sale_price"] <= 0]
  invalid_ppsf = df[df["median_sale_ppsf"] <= 0]
  if len(invalid_price) > 0 or len(invalid_ppsf) > 0:
    failed += 1
    print(f"Failed range check: {len(invalid_price):,} rows with price <= 0, {len(invalid_ppsf):,} rows with ppsf <= 0")
    raise ValueError("Range check failed, non-positive prices found.")
  else:
    passed += 1
    print(f"Passed range check: all prices and ppsf values are positive")
  
  
  # Duplicate check
  dupes = df.duplicated(subset=["region_name", "period_end", "duration"])
  if dupes.sum() > 0:
    failed += 1
    print(f"Failed duplicate check: {dupes.sum():,} duplicate (region_name, period_end) rows found")
    raise ValueError(f"Duplicate check failed. {dupes.sum()} duplicate rows found.")
  else:
    passed += 1
    print(f"Passed duplicate check: no duplicate (region_name, period_end) rows")
    
  
  # Row count check
  if len(df) < 1000:
    failed += 1
    print(f"  FAIL — Row count check: only {len(df):,} rows, expected at least 1,000")
    raise ValueError(f"Row count check failed. Only {len(df)} rows in DataFrame.")
  else:
    passed += 1
    print(f"Passed row count check: {len(df):,} rows")
    
  print(f"\nValidation complete: {passed} passed, {failed} failed")
  return True



# Standalone testing
if __name__ == "__main__":
    import pandas as pd
    from pipeline.transform import transform
 
    filename = "data/raw/weekly_housing_market_data_most_recent.tsv000"
    df_raw = pd.read_csv(filename, sep="\t")
    df_clean, mapping = transform(df_raw)
 
    result = validate(df_clean)
    print(f"\nValidation result: {result}")