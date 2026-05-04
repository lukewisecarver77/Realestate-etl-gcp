# Architecture Notes

## Pipeline Flow

```
Redfin Public S3
      ↓  (HTTP download)
extract.py  →  data/raw/metro_market_tracker.tsv
      ↓
transform.py  →  data/processed/clean_market_data.csv
      ↓
validate.py  →  raises on failure, passes silently
      ↓
load.py  →  BigQuery: {project}.realestate.metro_market
      ↓
regression.py + clustering.py  →  trained sklearn models
      ↓
plots.py  →  outputs/figures/*.png
```

## BigQuery Schema

| Column | Type | Description |
|---|---|---|
| region | STRING | Metro area name |
| period_end | DATE | Week ending date |
| median_sale_price | FLOAT | Median sale price USD |
| median_square_feet | FLOAT | Median home size |
| price_per_sqft | FLOAT | Derived: price / sqft |
| median_dom | FLOAT | Median days on market |
| inventory | FLOAT | Active listings count |
| price_tier | STRING | low / mid / high |
| dom_bucket | STRING | Binned days on market |
| month | INTEGER | Month number (1–12) |
| cluster_label | INTEGER | K-Means cluster assignment |

## GCP Services Used

- **Cloud Storage** (optional): could store raw files here instead of local disk
- **BigQuery**: data warehouse, target for cleaned data
- **Cloud Scheduler** (future): trigger pipeline on a schedule
- **Cloud Composer / Airflow** (future): full DAG orchestration

## Local Development vs. Cloud

For local development, `gcloud auth application-default login` handles auth.
No service account key file is needed.

In a production deployment, you would attach a service account with:
  - roles/bigquery.dataEditor
  - roles/bigquery.jobUser
