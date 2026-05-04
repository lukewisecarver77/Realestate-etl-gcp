import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")

FIGURES_PATH = "outputs/figures"


def _save(filename: str) -> None:
    os.makedirs(FIGURES_PATH, exist_ok=True)
    filepath = os.path.join(FIGURES_PATH, filename)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filepath}")


def plot_price_distribution(df: pd.DataFrame) -> None:
    """Histogram of median sale prices with mean line."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(df["median_sale_price"] / 1000, bins=60, color="#4C72B0", edgecolor="white", alpha=0.85)
    mean_price = df["median_sale_price"].mean() / 1000
    ax.axvline(mean_price, color="#C44E52", linestyle="--", linewidth=1.5, label=f"Mean: ${mean_price:,.0f}k")

    ax.set_title("Distribution of Median Sale Prices", fontsize=14, fontweight="bold")
    ax.set_xlabel("Median Sale Price ($000s)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.legend()

    _save("price_distribution.png")


def plot_cluster_scatter(df_clustered: pd.DataFrame) -> None:
    """Scatter plot of clusters: price vs days on market."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    cluster_names = {
        0: "Affordable Small Metros",
        1: "Distressed Markets",
        2: "High-End Metros",
        3: "Large City Markets",
    }

    for cluster_id in sorted(df_clustered["cluster_label"].unique()):
        subset = df_clustered[df_clustered["cluster_label"] == cluster_id].sample(
            min(2000, len(df_clustered[df_clustered["cluster_label"] == cluster_id])),
            random_state=42,
        )
        ax.scatter(
            subset["median_days_on_market"],
            subset["median_sale_price"] / 1000,
            c=colors[cluster_id],
            label=f"Cluster {cluster_id}: {cluster_names[cluster_id]}",
            alpha=0.4,
            s=10,
        )

    ax.set_title("Market Clusters: Price vs Days on Market", fontsize=14, fontweight="bold")
    ax.set_xlabel("Median Days on Market", fontsize=11)
    ax.set_ylabel("Median Sale Price ($000s)", fontsize=11)
    ax.legend(fontsize=9)

    _save("cluster_scatter.png")


def plot_elbow_curve(k_values: list, inertia_values: list, optimal_k: int = 4) -> None:
    """Line plot of K-Means inertia by number of clusters."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(k_values, inertia_values, marker="o", color="#4C72B0", linewidth=2)
    ax.axvline(optimal_k, color="#C44E52", linestyle="--", linewidth=1.5, label=f"Chosen K={optimal_k}")

    ax.set_title("Elbow Method — Optimal Number of Clusters", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Clusters (K)", fontsize=11)
    ax.set_ylabel("Inertia", fontsize=11)
    ax.legend()

    _save("elbow_curve.png")


def plot_feature_importance(feature_importances: dict) -> None:
    """Horizontal bar chart of Random Forest feature importances."""
    sorted_items = sorted(feature_importances.items(), key=lambda x: x[1])
    features = [item[0] for item in sorted_items]
    importances = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(features, importances, color="#4C72B0", edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)

    ax.set_title("Random Forest Feature Importances", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_xlim(0, max(importances) * 1.15)

    _save("feature_importance.png")


def plot_residuals(y_true, y_pred) -> None:
    """Scatter plot of residuals vs predicted values."""
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(y_pred / 1000, residuals / 1000, alpha=0.2, s=5, color="#4C72B0")
    ax.axhline(0, color="#C44E52", linestyle="--", linewidth=1.5)

    ax.set_title("Regression Residuals", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Price ($000s)", fontsize=11)
    ax.set_ylabel("Residual ($000s)", fontsize=11)

    _save("residuals.png")


def plot_cluster_heatmap(df_clustered: pd.DataFrame) -> None:
    """Seaborn heatmap of mean feature values per cluster."""
    cluster_names = {
        0: "Affordable Small",
        1: "Distressed",
        2: "High-End",
        3: "Large City",
    }

    summary = df_clustered.groupby("cluster_label")[[
        "median_sale_price",
        "price_per_sqft",
        "median_days_on_market",
        "active_listings",
        "percent_active_listings_with_price_drops",
    ]].mean()

    summary.index = [cluster_names[i] for i in summary.index]

    # Normalize each column for heatmap readability
    summary_norm = (summary - summary.min()) / (summary.max() - summary.min())

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(
        summary_norm,
        annot=summary.round(1),
        fmt="g",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
    )

    ax.set_title("Cluster Profiles (normalized, annotated with raw means)", fontsize=13, fontweight="bold")
    plt.xticks(rotation=20, ha="right")

    _save("cluster_heatmap.png")


# --- Standalone testing ---
if __name__ == "__main__":
    from pipeline.transform import transform
    from ml.regression import train_regression
    from ml.clustering import train_clustering

    filename = "data/raw/weekly_housing_market_data_most_recent.tsv000"
    df_raw = __import__("pandas").read_csv(filename, sep="\t")
    df_clean, _ = transform(df_raw)

    # Regression plots
    rf_model, lr_model, X_test, y_test, rf_preds, lr_preds, importances = train_regression(df_clean)
    plot_price_distribution(df_clean)
    plot_feature_importance(importances)
    plot_residuals(y_test, rf_preds)

    # Clustering plots
    df_clustered, model, scaler, k_vals, inertias, metrics = train_clustering(df_clean)
    plot_cluster_scatter(df_clustered)
    plot_elbow_curve(k_vals, inertias, optimal_k=4)
    plot_cluster_heatmap(df_clustered)

    print("\nAll charts saved to outputs/figures/")