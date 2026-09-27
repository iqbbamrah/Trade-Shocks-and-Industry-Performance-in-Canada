"""Gold layer: analysis tables for the research questions.

Mostly the DataFrame API; `trade_exposure` is written in Spark SQL to show the
same engine behind both (compare their `explain()` plans; they're equivalent).
"""
from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from .clean import QUARTILES
from .config import FIRST_YEAR, LAST_YEAR


def gdp_growth(gdp: DataFrame) -> DataFrame:
    """Q1: real GDP and year-over-year growth, by province x sector x year.

    lag() over a window partitioned by province x sector and ordered by year
    gives each row the previous year's value within the same series.
    """
    by_series = Window.partitionBy("province", "sector").orderBy("year")
    return (
        gdp.filter(F.col("year").between(FIRST_YEAR, LAST_YEAR))
        .withColumn(
            "real_growth_pct",
            # try_divide: a province-sector can report 0 GDP (e.g. in the territories).
            100 * (F.try_divide(F.col("real_gdp_m"), F.lag("real_gdp_m").over(by_series)) - 1),
        )
        .select("year", "province", "sector", "industry", "real_gdp_m", "real_growth_pct")
    )


def industry_cagr(growth: DataFrame) -> DataFrame:
    """Q1: compound annual growth rate of real GDP over the whole period.

    min_by/max_by pick the value at the earliest/latest year in one
    aggregation, with no self-join or window needed.
    """
    return (
        growth.groupBy("province", "sector", "industry")
        .agg(
            F.min("year").alias("first_year"),
            F.max("year").alias("last_year"),
            F.min_by("real_gdp_m", "year").alias("first_real_gdp_m"),
            F.max_by("real_gdp_m", "year").alias("last_real_gdp_m"),
        )
        .withColumn(
            "cagr_pct",
            F.when(
                (F.col("first_real_gdp_m") > 0) & (F.col("last_real_gdp_m") > 0),
                100 * (F.pow(F.col("last_real_gdp_m") / F.col("first_real_gdp_m"),
                             1 / (F.col("last_year") - F.col("first_year"))) - 1),
            ),
        )
    )


def trade_exposure(spark: SparkSession, trade: DataFrame, gdp: DataFrame) -> DataFrame:
    """Q2/Q3: national trade exposure and trade shocks by sector x year.

    trade_shock = change in log total trade. It's computed over the full trade
    history *before* restricting to the analysis window, so 2013 has a
    2012 -> 2013 shock instead of a null.

    The exposure ratios divide nominal trade by real (chained 2017 dollar)
    GDP, the only GDP series in the data. They're comparable across sectors
    and years, but aren't a strict current-dollar share.
    """
    trade.createOrReplaceTempView("silver_trade")
    gdp.createOrReplaceTempView("silver_gdp")
    return spark.sql(f"""
        WITH trade_m AS (
            SELECT
                year, sector, industry, n_unreliable_cells,
                exports_value_k / 1000 AS exports_m,
                imports_value_k / 1000 AS imports_m,
                (exports_value_k + imports_value_k) / 1000 AS total_trade_m
            FROM silver_trade
        ),
        with_shock AS (
            SELECT
                *,
                LN(total_trade_m) - LAG(LN(total_trade_m)) OVER (PARTITION BY sector ORDER BY year) AS trade_shock
            FROM trade_m
        ),
        national_gdp AS (
            SELECT year, sector, SUM(real_gdp_m) AS real_gdp_m
            FROM silver_gdp
            GROUP BY year, sector
        )
        SELECT
            s.year, s.sector, s.industry,
            s.exports_m, s.imports_m, s.total_trade_m,
            s.exports_m - s.imports_m                                  AS net_exports_m,
            g.real_gdp_m,
            100 * TRY_DIVIDE(s.total_trade_m, g.real_gdp_m)             AS trade_pct_gdp,
            100 * TRY_DIVIDE(s.exports_m - s.imports_m, g.real_gdp_m) AS net_exposure_pct_gdp,
            s.trade_shock,
            s.n_unreliable_cells
        FROM with_shock s
        JOIN national_gdp g USING (year, sector)
        WHERE s.year BETWEEN {FIRST_YEAR} AND {LAST_YEAR}
    """)


def firm_panel(revenue: DataFrame, exposure: DataFrame, fx: DataFrame) -> DataFrame:
    """Q3/Q4/Q5: national revenue panel, 4-digit NAICS industry x size cohort x year.

    Growth is only computed between consecutive years (an industry missing a
    year gets a null rather than a multi-year change labelled as one year).
    """
    entity = Window.partitionBy("naics", "size_cohort").orderBy("year")
    fx_changes = fx.withColumn(
        "fx_change", F.log("cad_per_usd") - F.lag(F.log("cad_per_usd")).over(Window.orderBy("year"))
    )
    return (
        revenue.filter(
            (F.col("geo_level") == "national")
            & (F.col("naics_level") == 4)
            & F.col("year").between(FIRST_YEAR, LAST_YEAR)
        )
        .select(
            "year", "naics", "sector", "industry", "size_cohort", "avg_revenue_k",
            F.log("avg_revenue_k").alias("log_revenue"),  # null for non-positive revenue
            (F.col("size_cohort") == "small").cast("int").alias("is_small"),
        )
        .withColumn(
            "revenue_growth",
            F.when(
                F.col("year") - F.lag("year").over(entity) == 1,
                F.col("log_revenue") - F.lag("log_revenue").over(entity),
            ),
        )
        .join(
            F.broadcast(exposure.select("year", "sector", "trade_shock", "trade_pct_gdp", "net_exposure_pct_gdp")),
            ["year", "sector"],
            "left",
        )
        .join(F.broadcast(fx_changes), "year", "left")
        .withColumn("shock_x_small", F.col("trade_shock") * F.col("is_small"))
    )


def fx_revenue_correlation(panel: DataFrame) -> DataFrame:
    """Q4: correlation between revenue growth and the CAD/USD change, by sector and size cohort."""
    return (
        panel.filter(F.col("revenue_growth").isNotNull() & F.col("fx_change").isNotNull())
        .groupBy("sector", "size_cohort")
        .agg(
            F.corr("revenue_growth", "fx_change").alias("corr_revenue_growth_fx"),
            F.count(F.lit(1)).alias("n_observations"),
        )
    )


def revenue_distribution(quartiles: DataFrame) -> DataFrame:
    """How unequal revenue is within each 2-digit industry: top vs. bottom quartile.

    The long quartile table is pivoted back to one column per quartile, which
    turns "top / bottom" into a plain column expression.
    """
    labels = list(QUARTILES.values())
    return (
        quartiles.filter((F.col("geo_level") == "national") & (F.col("naics_level") == 2))
        .groupBy("year", "naics", "industry", "size_cohort")
        .pivot("quartile", labels)
        .agg(F.first("avg_revenue_k"))
        # Spark 4 runs in ANSI mode, where x / 0 raises an error instead of
        # returning null; some industries report a bottom-quartile revenue of 0.
        .withColumn("top_to_bottom_ratio", F.try_divide("top", "bottom"))
    )
