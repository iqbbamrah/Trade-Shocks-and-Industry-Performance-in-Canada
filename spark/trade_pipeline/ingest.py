"""Bronze layer: raw CSVs -> typed Parquet, no business logic.

Every file is read with an explicit schema rather than `inferSchema=True`:
inference costs an extra full pass over the data, and it guesses. It would, for
example, read NAICS industry codes as integers, although they're identifiers
("31-33" in other files). `enforceSchema=False` makes Spark check the schema's
column names against each file's header, so a reordered or renamed column
fails loudly instead of silently shifting data into the wrong column.
"""
from __future__ import annotations

import re

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DataType, DoubleType, IntegerType, StringType, StructField, StructType

from . import config
from .io import ensure_unzipped, write_parquet

_REVENUE_MEASURES = [
    "Bottom quartile All other revenue", "Bottom quartile High revenue range",
    "Bottom quartile Low revenue range", "Bottom quartile Sales of goods and services",
    "Bottom quartile Total revenue",
    "Lower middle All other revenue", "Lower middle High revenue range",
    "Lower middle Low revenue range", "Lower middle Sales of goods and services",
    "Lower middle Total revenue",
    "Percent of businesses reporting all other revenues", "Percent of businesses reporting sales of goods",
    "Top quartile All other revenue", "Top quartile High revenue range",
    "Top quartile Low revenue range", "Top quartile Sales of goods and services",
    "Top quartile Total revenue",
    "Upper middle All other revenue", "Upper middle High revenue range",
    "Upper middle Low revenue range", "Upper middle Sales of goods and services",
    "Upper middle Total revenue",
    "Whole industry All other revenue", "Whole industry High revenue range",
    "Whole industry Sales of goods and services", "Whole industry Total revenue",
    "Whole industry low revenue range",
]

_EXPENSE_MEASURES = [
    "Cost of Sales (direct expenses)", "Bottom Quartile Cost of Sales", "Top Quartile Cost of Sales",
    "Wages and Benefits", "Bottom Quartile Wages and Benefits", "Top Quartile Wages and Benefits",
    "Purchases, materials and sub-contracts", "Bottom Quartile Purchases, materials and sub-contracts",
    "Top Quartile Purchases, materials and sub-contracts",
    "Bottom Quartile Closing Inventory", "Top Quartile Closing Inventory",
    "Bottom Quartile Operating Expenses (indirect)", "Top Quartile Operating Expenses (indirect)",
    "Bottom Quartile Labour and Commissions", "Top Quartile Labour and Commissions",
    "Interest and bank charges", "Bottom Quartile Interest and bank charges",
    "Top Quartile Interest and bank charges", "Professional and business fees",
    "Delivery, shipping and warehouse", "Bottom Quartile Delivery, shipping and warehouse expense",
    "Top Quartile Delivery, shipping and warehouse expense",
    "Bottom Quartile Other expenses", "Top Quartile Other expenses",
    "Total expenses", "Bottom Quartile Total expenses", "Top Quartile Total expenses",
    "Net Profit/Loss", "Bottom Quartile Net Profit/Loss", "Top Quartile Net Profit/Loss",
    "material_input_intensity", "implied_profit_margin", "logistics_exposure_index",
    "material_import_intensity",
]

_TRADE_COLUMNS: list[tuple[str, DataType]] = [
    ("REF_DATE", IntegerType()), ("GEO", StringType()), ("DGUID", StringType()),
    ("Estimates", StringType()), ("North American Industry Classification System (NAICS)", StringType()),
    ("UOM", StringType()), ("UOM_ID", IntegerType()), ("SCALAR_FACTOR", StringType()),
    ("SCALAR_ID", IntegerType()), ("VECTOR", StringType()), ("COORDINATE", StringType()),
    ("VALUE", DoubleType()), ("STATUS", StringType()), ("SYMBOL", StringType()),
    ("TERMINATED", StringType()), ("DECIMALS", IntegerType()),
]


def _schema(columns: list[tuple[str, DataType]]) -> StructType:
    return StructType([StructField(name, dtype, nullable=True) for name, dtype in columns])


SCHEMAS: dict[str, StructType] = {
    "revenue": _schema(
        [("Year", IntegerType()), ("NAICS", StringType()), ("Description_English", StringType()),
         ("Province", IntegerType()), ("Province_Description", StringType()),
         ("Incorporation Status", IntegerType())]
        + [(m, DoubleType()) for m in _REVENUE_MEASURES]
        + [("file_name", StringType())]
    ),
    "expenses": _schema(
        [("Year", IntegerType()), ("NAICS", StringType()), ("Description_English", StringType())]
        + [(m, DoubleType()) for m in _EXPENSE_MEASURES]
        + [("file_name", StringType())]
    ),
    "exports": _schema(_TRADE_COLUMNS),
    "imports": _schema(_TRADE_COLUMNS),
    # GDP values contain thousands separators ("30,873.00"), so they're read
    # as strings here and parsed in the silver layer.
    "gdp": _schema(
        [("Geography", StringType()), ("North American Industry Classification System (NAICS)", StringType()),
         ("REF-Date", IntegerType()), ("GDP", StringType())]
    ),
    "fx": _schema([("Year", IntegerType()), ("CAD_USD_Exchange_Rate", DoubleType())]),
}

SOURCE_FILES = {
    "revenue": "Revenue.csv",
    "expenses": "Expenses.csv",
    "exports": "export.csv",
    "imports": "import.csv",
    "gdp": "gdp.csv",
    "fx": "Exchange_Rate.csv",
}

# The two big tables are partitioned by year on disk, so a query filtered on
# year only reads the folders it needs (partition pruning).
PARTITION_COLUMNS = {"revenue": ["year"], "expenses": ["year"]}

# Tables whose meaning depends on row order (gdp.csv only names the province on
# the first row of each block). They get a `source_row` column at read time.
ORDER_SENSITIVE = {"gdp"}


def snake_case(name: str) -> str:
    """'Whole industry Total revenue' -> 'whole_industry_total_revenue'."""
    return re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_")


def read_raw(spark: SparkSession, table: str) -> DataFrame:
    path = ensure_unzipped(config.DATA_DIR / SOURCE_FILES[table])
    df = spark.read.csv(
        str(path),
        schema=SCHEMAS[table],
        header=True,
        enforceSchema=False,
        mode="FAILFAST",  # a value that doesn't fit its type is an error, not a silent null
    )
    df = df.toDF(*[snake_case(c) for c in df.columns])
    if table in ORDER_SENSITIVE:
        # Right after a file read, partitions follow file order, and
        # monotonically_increasing_id encodes (partition index, position in
        # partition), so it increases in file order. After any shuffle that
        # order is gone, which is why it's captured here, at the source.
        df = df.withColumn("source_row", F.monotonically_increasing_id())
    return df


def build_bronze(spark: SparkSession) -> dict[str, DataFrame]:
    tables = {}
    for table in SOURCE_FILES:
        df = read_raw(spark, table)
        write_parquet(df, config.BRONZE_DIR / table, PARTITION_COLUMNS.get(table, ()))
        tables[table] = df
    return tables
