"""
tests/test_validate.py — Unit Tests for Validation Stage

Use pytest. Run with: pytest tests/

Tests to implement:

  test_valid_dataframe_passes()
    - Build a clean, well-formed DataFrame
    - Assert validate() returns True without raising

  test_missing_column_raises()
    - Pass a DataFrame missing a required column
    - Assert validate() raises ValueError

  test_null_in_required_column_raises()
    - Pass a DataFrame with a null in a required column
    - Assert validate() raises ValueError

  test_negative_price_raises()
    - Pass a DataFrame with a row where median_sale_price <= 0
    - Assert validate() raises ValueError

  test_duplicate_rows_raises()
    - Pass a DataFrame with duplicate (region, date) rows
    - Assert validate() raises ValueError
"""
