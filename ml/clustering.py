import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from ml.evaluate import clustering_metrics
 
 
CLUSTER_FEATURES = [
    "median_sale_price",
    "price_per_sqft",
    "median_days_on_market",
    "active_listings",
    "percent_active_listings_with_price_drops",
]

def train_clustering(df: pd.DataFrame, random_state: int = 42) -> tuple:
  """
  Apply K-means clustering to grup metro areas into market segments.
  Uses the elbow method to find optimal K.
  Returns the labeled Dataframe, fitted model, scaler and intertia values
  """
  
  # Prepare clustering features
  
  df_cluster = df[CLUSTER_FEATURES + ["region_name_label"]].dropna()
  X = df_cluster[CLUSTER_FEATURES]
  X = X.replace([np.inf, -np.inf], np.nan)
  df_cluster = df_cluster[~X.isnull().any(axis=1)]
  X = df_cluster[CLUSTER_FEATURES]
  
  print(f"Clustering on {len(df_cluster):,} rows with {len(CLUSTER_FEATURES)} features")
  
  # Scale features 
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)
  
  # Elbow method for finding optimal K
  print("\nRunning elbow method: K=2 to K=10")
  k_values = range(2, 11)
  inertia_values = []
  
  for k in k_values:
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    km.fit(X_scaled)
    inertia_values.append(km.inertia_)
    print(f" K={k} inertia={km.inertia_:,.0f}")
    
  # Fit final model with chose K
  optimal_k = 4
  print(f"\nFitting final model with K={optimal_k}")
  final_model = KMeans(n_clusters=optimal_k, random_state=random_state, n_init=10)
  final_model.fit(X_scaled)
  labels = final_model.labels_
  
  # Add cluster labels back to DataFrame
  df_cluster = df_cluster.copy()
  df_cluster["cluster_label"] = labels
  
  # --- Evaluate ---
  print("\n--- Clustering Metrics ---")
  # Sample for silhouette score (too slow on full dataset)
  sample_size = 50000
  idx = np.random.choice(len(X_scaled), size=sample_size, replace=False)
  metrics = clustering_metrics(X_scaled[idx], labels[idx])
 
  # --- Cluster summary ---
  print("\n--- Cluster Summary (mean values) ---")
  summary = df_cluster.groupby("cluster_label")[CLUSTER_FEATURES].mean()
  print(summary.to_string())
 
  # --- Sample regions per cluster ---
  print("\n--- Sample Regions per Cluster ---")
  for cluster_id in range(optimal_k):
      sample = df_cluster[df_cluster["cluster_label"] == cluster_id]["region_name_label"].unique()[:5]
      print(f"  Cluster {cluster_id}: {list(sample)}")
 
  return df_cluster, final_model, scaler, list(k_values), inertia_values, metrics
 
 
# --- Standalone testing ---
if __name__ == "__main__":
    from pipeline.transform import transform
 
    filename = "data/raw/weekly_housing_market_data_most_recent.tsv000"
    df_raw = __import__("pandas").read_csv(filename, sep="\t")
    df_clean, _ = transform(df_raw)
 
    df_clustered, model, scaler, k_vals, inertias, metrics = train_clustering(df_clean)
    print("\nDone. Clustering complete.")
    print()