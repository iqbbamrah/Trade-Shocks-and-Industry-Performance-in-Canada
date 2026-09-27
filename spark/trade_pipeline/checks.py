"""Data-quality checks run between layers; the pipeline stops if any fail."""
from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


class DataQualityError(Exception):
    pass


def expect_unique(df: DataFrame, keys: list[str], table: str) -> str | None:
    n = df.groupBy(*keys).count().filter(F.col("count") > 1).count()
    return f"{table}: {n} duplicate keys on {keys}" if n else None


def expect_not_null(df: DataFrame, columns: list[str], table: str) -> str | None:
    counts = df.select([F.count_if(F.col(c).isNull()).alias(c) for c in columns]).first().asDict()
    bad = {c: n for c, n in counts.items() if n}
    return f"{table}: nulls in {bad}" if bad else None


def expect_min_rows(df: DataFrame, min_rows: int, table: str) -> str | None:
    n = df.count()
    return f"{table}: {n} rows, expected at least {min_rows}" if n < min_rows else None


def expect_between(df: DataFrame, column: str, low: float, high: float, table: str) -> str | None:
    n = df.filter(~F.col(column).between(low, high)).count()
    return f"{table}: {n} rows with {column} outside [{low}, {high}]" if n else None


def run(checks: list[str | None], layer: str) -> int:
    failures = [c for c in checks if c]
    if failures:
        raise DataQualityError(f"{layer} checks failed:\n  " + "\n  ".join(failures))
    return len(checks)


def silver_checks(t: dict[str, DataFrame]) -> list[str | None]:
    return [
        # 231,400 rows expected (full-detail records, provinces + national only).
        expect_min_rows(t["revenue"], 225_000, "revenue"),
        expect_unique(t["revenue"], ["year", "naics", "geography", "size_cohort"], "revenue"),
        expect_not_null(t["revenue"], ["year", "naics", "sector", "geo_level", "size_cohort"], "revenue"),
        expect_unique(t["revenue_quartiles"], ["year", "naics", "geography", "size_cohort", "quartile"], "revenue_quartiles"),
        expect_unique(t["expenses"], ["year", "naics", "size_cohort"], "expenses"),
        expect_not_null(t["expenses"], ["year", "naics", "size_cohort"], "expenses"),
        expect_unique(t["trade"], ["year", "sector"], "trade"),
        expect_unique(t["gdp"], ["year", "province", "sector"], "gdp"),
        expect_not_null(t["gdp"], ["year", "province", "sector", "real_gdp_m"], "gdp"),
        expect_unique(t["fx"], ["year"], "fx"),
    ]


def gold_checks(t: dict[str, DataFrame]) -> list[str | None]:
    return [
        expect_unique(t["gdp_growth"], ["year", "province", "sector"], "gdp_growth"),
        expect_unique(t["industry_cagr"], ["province", "sector"], "industry_cagr"),
        expect_unique(t["trade_exposure"], ["year", "sector"], "trade_exposure"),
        expect_between(t["trade_exposure"], "trade_pct_gdp", 0, 10_000, "trade_exposure"),
        expect_unique(t["firm_panel"], ["year", "naics", "size_cohort"], "firm_panel"),
        expect_min_rows(t["firm_panel"], 1_000, "firm_panel"),
        expect_not_null(t["firm_panel"], ["trade_shock"], "firm_panel"),
    ]
