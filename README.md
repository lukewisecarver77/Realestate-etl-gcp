# Real Estate ETL Pipeline — GCP

A cloud-native data engineering pipeline that extracts public real estate data, transforms and loads it into Google BigQuery, and applies machine learning to predict property values and identify regional market segments.

Built as a portfolio project demonstrating end-to-end data engineering on Google Cloud Platform.

---

## Architecture Overview

```
[Redfin CSV Download]
        ↓
   extract.py         ← Download and ingest raw data
        ↓
  transform.py        ← Clean, normalize, engineer features
        ↓
   validate.py        ← Data quality checks before load
        ↓
    load.py           ← Load clean data into BigQuery
        ↓
  regression.py       ← Scikit-learn price prediction model
  clustering.py       ← K-Means regional market segmentation
        ↓
    plots.py          ← Matplotlib/Seaborn visualizations
        ↓
  outputs/figures/    ← Saved charts and reports
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.11 |
| Data Processing | Pandas, NumPy |
| Cloud Warehouse | Google BigQuery |
| ML | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Version Control | Git / GitHub |
| Config | python-dotenv |

---

## Project Structure

```
realestate-etl-gcp/
├── data/
│   ├── raw/                  # Raw downloaded CSVs (gitignored)
│   └── processed/            # Cleaned data before BigQuery load (gitignored)
├── pipeline/
│   ├── extract.py            # Data download and ingestion
│   ├── transform.py          # Cleaning, normalization, feature engineering
│   ├── validate.py           # Pre-load data quality checks
│   └── load.py               # BigQuery load logic
├── ml/
│   ├── regression.py         # Price prediction model (Linear Regression / Random Forest)
│   ├── clustering.py         # K-Means market segmentation
│   └── evaluate.py           # Model evaluation utilities (RMSE, R², silhouette score)
├── visualization/
│   └── plots.py              # All Matplotlib and Seaborn chart generation
├── tests/
│   ├── test_transform.py     # Unit tests for transform logic
│   └── test_validate.py      # Unit tests for validation checks
├── docs/
│   └── architecture.md       # Detailed architecture notes
├── outputs/
│   └── figures/              # Generated charts saved here
├── main.py                   # Pipeline entrypoint — runs full workflow
├── config.py                 # Centralized config and constants
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
└── .gitignore
```

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/realestate-etl-gcp.git
cd realestate-etl-gcp
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
# Edit .env with your GCP project ID and BigQuery dataset name
```

### 5. Authenticate with GCP
```bash
gcloud auth application-default login
```

### 6. Run the full pipeline
```bash
python main.py
```

---

## Data Source

This project uses publicly available housing market data from the [Redfin Data Center](https://www.redfin.com/news/data-center/). No API key required — data is downloadable as CSV.

Relevant files:
- `metro_market_tracker.tsv000` — Metro-level weekly housing stats
- State or zip-level files for more granular analysis

---

## Pipeline Stages

### Extract (`pipeline/extract.py`)
- Downloads the Redfin CSV/TSV data file
- Saves raw file to `data/raw/`
- Logs file size and row count on ingest

### Transform (`pipeline/transform.py`)
- Drops nulls and irrelevant columns
- Normalizes column names (lowercase, underscores)
- Parses and standardizes date fields
- Engineers derived features:
  - `price_per_sqft` = median sale price / median square footage
  - `days_on_market_bucket` = binned DOM categories
  - `price_tier` = low / mid / high based on percentile
- Outputs clean DataFrame to `data/processed/`

### Validate (`pipeline/validate.py`)
- Checks for null values in required columns
- Validates value ranges (e.g. price > 0, sqft > 0)
- Confirms expected column schema is present
- Raises exceptions and logs warnings before any data reaches BigQuery

### Load (`pipeline/load.py`)
- Authenticates with GCP using application default credentials
- Creates BigQuery dataset if it doesn't exist
- Loads clean DataFrame into a target BigQuery table
- Supports WRITE_TRUNCATE or WRITE_APPEND mode via config flag

---

## Machine Learning

### Price Prediction (`ml/regression.py`)
Trains a regression model to predict median sale price from property features.

Features used:
- Median square footage
- Region / metro area (encoded)
- Inventory levels
- Days on market
- Season / month

Models tried:
- Linear Regression (baseline)
- Random Forest Regressor (primary)

### Market Segmentation (`ml/clustering.py`)
Applies K-Means clustering to group metro areas into market segments based on:
- Median sale price
- Price per square footage
- Days on market
- Inventory levels
- Price change trends

Optimal K determined via elbow method.

### Evaluation (`ml/evaluate.py`)
- Regression: RMSE, MAE, R²
- Clustering: Inertia curve (elbow), Silhouette score
- Train/test split: 80/20

---

## Visualizations

All charts saved to `outputs/figures/`.

| Chart | Description |
|---|---|
| `price_distribution.png` | Histogram of median sale prices across dataset |
| `cluster_scatter.png` | Scatter plot of clusters: price vs. days on market |
| `cluster_map.png` | Cluster labels by metro region |
| `feature_importance.png` | Random Forest feature importances |
| `elbow_curve.png` | K-Means inertia by number of clusters |
| `residuals.png` | Regression residual plot |

---

## Results

## Regression
- **Regression R²**: 0.8839
- **Regression RMSE**: $64,368
- **Regression MAE**: $31,492
- **Top Feature**: region_name_encoded (0.4802) — location drives price more than any other factor

## Clustering
- **Optimal Clusters (K)**: 4
- **Silhouette Score**: 0.5909
- **Cluster 0**: Affordable small metros (~$242k)
- **Cluster 1**: Distressed markets with high price drop rate
- **Cluster 2**: High-end lifestyle metros (~$755k)
- **Cluster 3**: Large high-inventory city markets (~$377k)
---

## Future Improvements
- Automate ingestion with GCP Cloud Scheduler or Cloud Composer (Airflow)
- Add a Dataflow job for a streaming pipeline variant
- Store model artifacts in GCP Cloud Storage
- Build a Looker Studio dashboard on top of the BigQuery table

---

## Author

James "Luke" Wisecarver
[GitHub](https://github.com/lukewisecarver77) | [LinkedIn](https://www.linkedin.com/in/luke-wisecarver-762792240/)
