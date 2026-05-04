import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import silhouette_score

def regression_metrics(y_true, y_pred) -> dict:
  """
  Compute and print Root Mean Squared Error, Mean Absolute Error, and R2 for regression predictions
  Return a dictionary of the metrics.
  """
  
  rmse = np.sqrt(mean_squared_error(y_true, y_pred))
  mae = mean_absolute_error(y_true, y_pred)
  r2 = r2_score(y_true, y_pred)
  
  print(f"RMSE: ${rmse:,.0f}")
  print(f"MAE: ${mae:,.0f}")
  print(f" R2: {r2:.4f}")
  
  return{"rmse": rmse, "mae":mae, "r2": r2}


def clustering_metrics(X_scaled, labels) -> dict:
  """
  Compute and print silhouette score for clustering results.
  Returns a dictionary of the metrics
  """
  
  score = silhouette_score(X_scaled, labels)
  print(f"Silhouette Score: {score:.4f}")
  
  return {"Silhouette_score": score}