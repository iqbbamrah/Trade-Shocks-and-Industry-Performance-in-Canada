"""Silver layer: cleaned, deduplicated, reshaped tables.

Each function takes bronze DataFrames and returns a silver DataFrame, with no
I/O, so they can be unit-tested on tiny hand-built inputs (see tests/).
"""
from __future__ import annotations

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

PROVINCES = [
    "Newfoundland and Labrador", "Prince Edward Island", "Nova Scotia", "New Brunswick",
    "Quebec", "Ontario", "Manitoba", "Saskatchewan", "Alberta", "British Columbia",
    "Yukon", "Northwest Territories", "Nunavut",
]

# The financial files come in two firm-size cohorts, identified only by file name.
SIZE_COHORTS = {
    "revenue30K": "small", "expenses30K": "small",  # $30K-$5M in annual revenue
    "revenue5M": "large", "expenses5M": "large",     # $5M-$20M
}

QUARTILES = {
    "bottom_quartile": "bottom", "lower_middle": "lower_middle",
    "upper_middle": "upper_middle", "top_quartile": "top",
}

EXPENSE_MEASURES = {
    "total_expenses": "total_expenses_k",
    "net_profit_loss": "net_profit_k",
    "cost_of_sales_direct_expenses": "cost_of_sales_k",
    "wages_and_benefits": "wages_and_benefits_k",
    "implied_profit_margin": "profit_margin",
}


def naics_sector(naics: Column) -> Column:
    """2-digit NAICS sector, using the combined codes the trade and GDP tables use."""
    two = F.substring(naics, 1, 2)
    return (
        F.when(two.isin("31", "32", "33"), F.lit("31-33"))
        .when(two.isin("44", "45"), F.lit("44-45"))
        .when(two.isin("48", "49"), F.lit("48-49"))
        .otherwise(two)
    )


def bracket_code(label: Column) -> Column:
    """'Manufacturing  [31-33]' -> '31-33'; totals ('All industries', '[T001]') -> 'TOTAL'."""
    code = F.regexp_extract(label, r"\[([^\]]+)\]", 1)
    return F.when((code == "") | code.startswith("T"), F.lit("TOTAL")).otherwise(code)


def strip_bracket(label: Column) -> Column:
    """'Manufacturing  [31-33]' -> 'Manufacturing'."""
    return F.trim(F.regexp_replace(label, r"\s*\[[^\]]*\].*$", ""))


def _size_cohort(file_name: Column) -> Column:
    mapping = F.create_map(*[F.lit(x) for kv in SIZE_COHORTS.items() for x in kv])
    return mapping[file_name]


def clean_revenue(bronze: DataFrame) -> DataFrame:
    """One row per year x NAICS x geography x size cohort.

    The export repeats keys: 67K rows are exact duplicates, and after removing
    them most keys still have 2-3 records for sub-populations the export doesn't
    label. Exactly one record per key carries the full detail (the sales-of-goods
    / other-revenue breakdown), so that's kept as the canonical row. Summing
    across the records instead would add up per-business averages from
    overlapping populations.
    """
    geo_level = (
        F.when(F.col("province_description") == "Canada Total", F.lit("national"))
        .when(F.col("province_description").isin(PROVINCES), F.lit("province"))
    )
    return (
        bronze.dropDuplicates()
        .filter(F.col("whole_industry_sales_of_goods_and_services").isNotNull())
        .withColumn("geo_level", geo_level)
        # Drop aggregates like "Canada excluding Alberta" and the one malformed 1-digit code.
        .filter(F.col("geo_level").isNotNull() & (F.length("naics") >= 2))
        .select(
            "year",
            "naics",
            F.length("naics").alias("naics_level"),
            naics_sector(F.col("naics")).alias("sector"),
            F.col("description_english").alias("industry"),
            "geo_level",
            F.col("province_description").alias("geography"),
            _size_cohort(F.col("file_name")).alias("size_cohort"),
            # All revenue figures are per-business averages, in $ thousands.
            F.col("whole_industry_total_revenue").alias("avg_revenue_k"),
            F.col("whole_industry_sales_of_goods_and_services").alias("avg_sales_of_goods_k"),
            F.col("whole_industry_all_other_revenue").alias("avg_other_revenue_k"),
            F.col("percent_of_businesses_reporting_sales_of_goods").alias("pct_reporting_sales_of_goods"),
            *[
                F.col(f"{prefix}_{measure}")
                for prefix in QUARTILES
                for measure in ("total_revenue", "low_revenue_range", "high_revenue_range")
            ],
        )
    )


def unpivot_revenue_quartiles(revenue: DataFrame) -> DataFrame:
    """Wide -> long: one row per revenue quartile instead of 4 x 3 quartile columns.

    Each row gets an array with one struct per quartile, and `inline` explodes
    the array into rows and the struct fields into columns. (DataFrame.unpivot
    only handles a single value column; this handles several at once.)
    """
    quartile_structs = F.array(*[
        F.struct(
            F.lit(label).alias("quartile"),
            F.col(f"{prefix}_total_revenue").alias("avg_revenue_k"),
            F.col(f"{prefix}_low_revenue_range").alias("revenue_range_low_k"),
            F.col(f"{prefix}_high_revenue_range").alias("revenue_range_high_k"),
        )
        for prefix, label in QUARTILES.items()
    ])
    return revenue.select(
        "year", "naics", "naics_level", "sector", "industry", "geo_level", "geography", "size_cohort",
        F.inline(quartile_structs),
    )


def clean_expenses(bronze: DataFrame) -> DataFrame:
    """One row per year x NAICS x size cohort (the expense data is national only).

    After exact-duplicate removal, keys still have up to ~20 records whose
    distinguishing dimension isn't in the export, so no single record can be
    chosen. The median across them is used, with the record count kept so
    thinly-supported rows are visible.
    """
    return (
        bronze.dropDuplicates()
        .filter(F.length("naics") >= 2)
        .groupBy(
            "year",
            "naics",
            F.col("description_english").alias("industry"),
            _size_cohort(F.col("file_name")).alias("size_cohort"),
        )
        .agg(
            *[F.median(src).alias(dst) for src, dst in EXPENSE_MEASURES.items()],
            F.count(F.lit(1)).alias("n_source_records"),
        )
        .withColumn("naics_level", F.length("naics"))
        .withColumn("sector", naics_sector(F.col("naics")))
    )


def clean_trade(exports: DataFrame, imports: DataFrame) -> DataFrame:
    """National trade by sector and year: one row per year x sector.

    The source is StatCan's long format, one row per measure. Rows are
    stacked (union), labelled by flow and measure, then pivoted back out to one
    column per flow x measure.
    """
    naics_label = F.col("north_american_industry_classification_system_naics")
    stacked = exports.withColumn("flow", F.lit("exports")).unionByName(
        imports.withColumn("flow", F.lit("imports"))
    )
    series = F.concat_ws(
        "_",
        F.col("flow"),
        F.when(F.col("estimates").startswith("Value"), F.lit("value_k")).otherwise(F.lit("establishments")),
    )
    return (
        stacked.filter(F.col("geo") == "Canada")
        .select(
            F.col("ref_date").alias("year"),
            bracket_code(naics_label).alias("sector"),
            strip_bracket(naics_label).alias("industry"),
            series.alias("series"),
            # '..' = not available. 'F' (too unreliable to publish) is counted
            # separately below so downstream users can see how much of a value
            # rests on flagged cells.
            F.when(F.col("status") == "..", F.lit(None)).otherwise(F.col("value")).alias("value"),
            (F.col("status") == "F").cast("int").alias("is_unreliable"),
        )
        .groupBy("year", "sector", "industry")
        .pivot("series", ["exports_value_k", "exports_establishments", "imports_value_k", "imports_establishments"])
        .agg(F.first("value"))
        .join(
            stacked.filter(F.col("geo") == "Canada")
            .groupBy(F.col("ref_date").alias("year"), bracket_code(naics_label).alias("sector"))
            .agg(F.sum((F.col("status") == "F").cast("int")).alias("n_unreliable_cells")),
            ["year", "sector"],
        )
    )


def clean_gdp(bronze: DataFrame) -> DataFrame:
    """Province x sector x year real GDP ($ millions, chained 2017 dollars).

    The series is already inflation-adjusted: provinces sum to ~$1.84T (2013)
    and ~$2.25T (2023), ~2%/yr, matching Canada's real (not nominal) growth.
    Deflating it by CPI again would count inflation twice.

    The export only names the province and industry on the first row of each
    block, so they have to be filled down. Spark DataFrames have no inherent
    row order, so this relies on `source_row`, captured at ingestion while the
    file order was still known, and fills with `last(..., ignorenulls=True)`
    over a window running from the first row to the current one. (A window with
    no partitionBy runs in one task, which is fine for this 3K-row table.)
    """
    running = Window.orderBy("source_row").rowsBetween(Window.unboundedPreceding, Window.currentRow)
    naics_label = F.last("north_american_industry_classification_system_naics", ignorenulls=True).over(running)
    return (
        bronze.select(
            F.col("ref_date").alias("year"),
            F.last("geography", ignorenulls=True).over(running).alias("province"),
            bracket_code(naics_label).alias("sector"),
            strip_bracket(naics_label).alias("industry"),
            F.regexp_replace("gdp", ",", "").cast("double").alias("real_gdp_m"),
        )
    )


def clean_fx(bronze: DataFrame) -> DataFrame:
    return bronze.select("year", F.col("cad_usd_exchange_rate").alias("cad_per_usd"))
