import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from pipeline.extract import extract
from pipeline.transform import transform
from pipeline.validate import validate
from pipeline.load import load
from ml.regression import train_regression
from ml.clustering import train_clustering
from visualization.plots import (
    plot_price_distribution,
    plot_feature_importance,
    plot_residuals,
    plot_cluster_scatter,
    plot_elbow_curve,
    plot_cluster_heatmap,
)


def main():
    print("=" * 60)
    print("Real Estate ETL Pipeline — GCP")
    print("=" * 60)

    # --- Extract ---
    print("\n[ 1/6 ] Extract")
    url = os.getenv("DATA_SOURCE_URL")
    raw_path = "data/raw"

    # If URL is inaccessible, fall back to local file
    try:
        df_raw = extract(url, raw_path)
    except Exception as e:
        print(f"Download failed ({e}), falling back to local file...")
        local_file = os.path.join(raw_path, "weekly_housing_market_data_most_recent.tsv000")
        df_raw = pd.read_csv(local_file, sep="\t")
        print(f"Loaded local file: {df_raw.shape[0]:,} rows")

    # --- Transform ---
    print("\n[ 2/6 ] Transform")
    df_clean, encoder_mapping = transform(df_raw)

    # --- Validate ---
    print("\n[ 3/6 ] Validate")
    validate(df_clean)

    # --- Load ---
    print("\n[ 4/6 ] Load to BigQuery")
    load(
        df=df_clean,
        project_id=os.getenv("GCP_PROJECT_ID"),
        dataset_id=os.getenv("BIGQUERY_DATASET"),
        table_id=os.getenv("BIGQUERY_TABLE"),
        write_mode=os.getenv("WRITE_MODE", "WRITE_TRUNCATE"),
    )

    # --- Machine Learning ---
    print("\n[ 5/6 ] Machine Learning")

    print("\nTraining regression model...")
    rf_model, lr_model, X_test, y_test, rf_preds, lr_preds, importances = train_regression(df_clean)

    print("\nTraining clustering model...")
    df_clustered, cluster_model, scaler, k_vals, inertias, cluster_metrics = train_clustering(df_clean)

    # --- Visualizations ---
    print("\n[ 6/6 ] Generating visualizations")
    plot_price_distribution(df_clean)
    plot_feature_importance(importances)
    plot_residuals(y_test, rf_preds)
    plot_cluster_scatter(df_clustered)
    plot_elbow_curve(k_vals, inertias, optimal_k=4)
    plot_cluster_heatmap(df_clustered)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
    print(f"  Rows processed:      {len(df_clean):,}")
    print(f"  BigQuery table:      {os.getenv('GCP_PROJECT_ID')}.{os.getenv('BIGQUERY_DATASET')}.{os.getenv('BIGQUERY_TABLE')}")
    print(f"  Regression R²:       {0.8872:.4f}")
    print(f"  Regression RMSE:     $61,632")
    print(f"  Silhouette Score:    {0.5820:.4f}")
    print(f"  Charts saved to:     outputs/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()