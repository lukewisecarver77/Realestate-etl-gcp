import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ml.evaluate import regression_metrics
 
 
FEATURE_COLS = [
    "median_days_on_market",
    "active_listings",
    "median_pending_sqft",
    "average_sale_to_list_ratio",
    "percent_active_listings_with_price_drops",
    "weeks_of_supply",
    "month",
    "region_name_encoded",
]

TARGET_COL = "median_sale_price"

def train_regression(df: pd.DataFrame, random_state: int = 42) -> tuple:
  """
  Train a linear regression baseline and a random forest model
  to predict median_sale_price. Returns both trained models, the test set and feature importances
  """
  
  # Prepare features and target
  df_model = df[FEATURE_COLS + [TARGET_COL]].dropna()
  X = df_model[FEATURE_COLS]
  y = df_model[TARGET_COL]
  
  print(f"Training on {len(df_model):,} rows with {len(FEATURE_COLS)} features")
  
  # Train/test split
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=random_state
  )
    
  # Baseline: Linear Regression
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)
    
  lr = LinearRegression()
  lr.fit(X_train_scaled, y_train)
  lr_preds = lr.predict(X_test_scaled)
  
  print("\n--- Linear Regression (Baseline) ---")
  lr_metrics = regression_metrics(y_test, lr_preds)
    
  # --- Primary: Random Forest ---
  rf = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
  rf.fit(X_train, y_train)
  rf_preds = rf.predict(X_test)
 
  print("\n--- Random Forest ---")
  rf_metrics = regression_metrics(y_test, rf_preds)
 
  # --- Feature importances ---
  feature_importances = dict(zip(FEATURE_COLS, rf.feature_importances_))
  print("\n--- Feature Importances ---")
  for feat, imp in sorted(feature_importances.items(), key=lambda x: x[1], reverse=True):
    print(f"  {feat:<45} {imp:.4f}")
 
  return rf, lr, X_test, y_test, rf_preds, lr_preds, feature_importances
 
 
# --- Standalone testing ---
if __name__ == "__main__":
    import os
    from pipeline.transform import transform
 
    filename = "data/raw/weekly_housing_market_data_most_recent.tsv000"
    df_raw = __import__("pandas").read_csv(filename, sep="\t")
    df_clean, _ = transform(df_raw)
 
    rf_model, lr_model, X_test, y_test, rf_preds, lr_preds, importances = train_regression(df_clean)
    print("\nDone. Models trained successfully.")
 