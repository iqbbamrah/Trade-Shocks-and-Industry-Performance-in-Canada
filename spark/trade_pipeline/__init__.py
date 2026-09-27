"""PySpark rebuild of the Trade Shocks data pipeline.

Layers (the "medallion" pattern common in Spark/Databricks projects):
    bronze  raw CSVs read with explicit schemas, saved as Parquet
    silver  cleaned, deduplicated, reshaped tables
    gold    analysis tables for the research questions (Q1-Q5)
    models  Spark ML regressions for Q3 and Q5
"""
