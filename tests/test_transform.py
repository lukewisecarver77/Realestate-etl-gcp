"""
tests/test_transform.py — Unit Tests for Transform Stage

Use pytest. Run with: pytest tests/

Tests to implement:

  test_column_names_normalized()
    - Pass a DataFrame with messy column names (spaces, caps)
    - Assert all output column names are lowercase with underscores

  test_nulls_dropped()
    - Pass a DataFrame with known null rows in required columns
    - Assert those rows are not present in output

  test_price_per_sqft_calculated()
    - Pass a DataFrame with known price and sqft values
    - Assert price_per_sqft = price / sqft within floating point tolerance

  test_price_tier_values()
    - Assert price_tier column only contains 'low', 'mid', 'high'

  test_dom_bucket_values()
    - Assert dom_bucket column only contains expected bin labels

  test_no_negative_prices()
    - Assert median_sale_price > 0 for all rows in output
"""
