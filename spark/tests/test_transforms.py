"""Unit tests for the silver/gold transformations, on tiny hand-built inputs."""
import math

from pyspark.sql import functions as F

from trade_pipeline import clean, metrics
from trade_pipeline.ingest import snake_case

QUARTILE_COLS = [
    f"{q}_{m}" for q in clean.QUARTILES for m in ("total_revenue", "low_revenue_range", "high_revenue_range")
]


def revenue_row(year=2019, naics="1111", geo="Canada Total", file_name="revenue30K", total=100.0, sales=90.0, **quartiles):
    row = {
        "year": year, "naics": naics, "description_english": "Oilseed and Grain Farming",
        "province_description": geo, "file_name": file_name,
        "whole_industry_total_revenue": total,
        "whole_industry_sales_of_goods_and_services": sales,
        "whole_industry_all_other_revenue": None if sales is None else total - sales,
        "percent_of_businesses_reporting_sales_of_goods": 95.0,
    }
    row.update({c: quartiles.get(c, 1.0) for c in QUARTILE_COLS})
    return row


def test_snake_case():
    assert snake_case("Whole industry Total revenue") == "whole_industry_total_revenue"
    assert snake_case("Net Profit/Loss") == "net_profit_loss"
    assert snake_case("North American Industry Classification System (NAICS)") == \
        "north_american_industry_classification_system_naics"


def test_naics_sector_and_bracket_codes(spark):
    df = spark.createDataFrame(
        [("3111", "Manufacturing  [31-33]"), ("4412", "All industries  [T001] 5"),
         ("4885", "All industries"), ("5211", "Finance and insurance [52]")],
        "naics string, label string",
    )
    rows = df.select(
        clean.naics_sector(F.col("naics")).alias("sector"),
        clean.bracket_code(F.col("label")).alias("code"),
        clean.strip_bracket(F.col("label")).alias("name"),
    ).collect()
    assert [r.sector for r in rows] == ["31-33", "44-45", "48-49", "52"]
    assert [r.code for r in rows] == ["31-33", "TOTAL", "TOTAL", "52"]
    assert [r.name for r in rows] == ["Manufacturing", "All industries", "All industries", "Finance and insurance"]


def test_clean_revenue_keeps_the_full_detail_record(spark):
    rows = [
        revenue_row(total=100.0, sales=90.0),               # canonical full-detail record
        revenue_row(total=100.0, sales=90.0),               # exact duplicate
        revenue_row(total=40.0, sales=None),                # unlabeled sub-population
        revenue_row(geo="Canada excluding Alberta"),        # aggregate geography
        revenue_row(naics="1"),                             # malformed code
        revenue_row(geo="Ontario", file_name="revenue5M"),  # a different key
    ]
    out = clean.clean_revenue(spark.createDataFrame(rows)).orderBy("geography").collect()
    assert len(out) == 2
    national, ontario = out
    assert (national.geo_level, national.avg_revenue_k, national.size_cohort) == ("national", 100.0, "small")
    assert (ontario.geo_level, ontario.size_cohort, ontario.sector) == ("province", "large", "11")


def test_unpivot_revenue_quartiles(spark):
    revenue = clean.clean_revenue(spark.createDataFrame([
        revenue_row(bottom_quartile_total_revenue=10.0, top_quartile_total_revenue=80.0, top_quartile_high_revenue_range=5000.0)
    ]))
    long = {r.quartile: r for r in clean.unpivot_revenue_quartiles(revenue).collect()}
    assert set(long) == {"bottom", "lower_middle", "upper_middle", "top"}
    assert long["bottom"].avg_revenue_k == 10.0
    assert (long["top"].avg_revenue_k, long["top"].revenue_range_high_k) == (80.0, 5000.0)


def test_clean_gdp_fills_down_in_file_order(spark):
    # Mirrors the export: province and industry only on the first row of each block.
    col = "north_american_industry_classification_system_naics"
    bronze = spark.createDataFrame(
        [("Ontario", "All industries  [T001] 5", 2013, "1,000.5", 0),
         (None, None, 2014, "1,100.0", 1),
         (None, "Utilities  [22]", 2013, "50.0", 2),
         ("Quebec", "All industries  [T001] 5", 2013, "900.0", 3),
         (None, "Utilities  [22]", 2013, "40.0", 4)],
        f"geography string, {col} string, ref_date int, gdp string, source_row long",
    ).orderBy(F.desc("source_row"))  # scrambled physical order: the fill must follow source_row
    got = {(r.province, r.sector, r.year): r.real_gdp_m for r in clean.clean_gdp(bronze).collect()}
    assert got == {
        ("Ontario", "TOTAL", 2013): 1000.5, ("Ontario", "TOTAL", 2014): 1100.0,
        ("Ontario", "22", 2013): 50.0,
        ("Quebec", "TOTAL", 2013): 900.0, ("Quebec", "22", 2013): 40.0,
    }


def test_clean_trade_pivots_and_handles_status_flags(spark):
    cols = "ref_date int, geo string, estimates string, north_american_industry_classification_system_naics string, value double, status string"
    exports = spark.createDataFrame([
        (2020, "Canada", "Value of exports", "Manufacturing [31-33]", 500.0, None),
        (2020, "Canada", "Number of exporting establishments", "Manufacturing [31-33]", 7.0, "F"),
        (2020, "Toronto, Ontario", "Value of exports", "Manufacturing [31-33]", 99.0, None),  # not national
    ], cols)
    imports = spark.createDataFrame([
        (2020, "Canada", "Value of imports", "Manufacturing [31-33]", 300.0, ".."),  # not available
        (2020, "Canada", "Number of importing establishments", "Manufacturing [31-33]", 5.0, None),
    ], cols)
    [row] = clean.clean_trade(exports, imports).collect()
    assert (row.sector, row.industry) == ("31-33", "Manufacturing")
    assert (row.exports_value_k, row.exports_establishments) == (500.0, 7.0)
    assert row.imports_value_k is None and row.imports_establishments == 5.0
    assert row.n_unreliable_cells == 1


def test_firm_panel_growth_skips_year_gaps(spark):
    revenue = spark.createDataFrame(
        [(2013, "1111", 4, "11", "Farming", "national", "small", 100.0),
         (2014, "1111", 4, "11", "Farming", "national", "small", 110.0),
         (2016, "1111", 4, "11", "Farming", "national", "small", 130.0)],  # 2015 missing
        "year int, naics string, naics_level int, sector string, industry string, geo_level string, size_cohort string, avg_revenue_k double",
    )
    exposure = spark.createDataFrame(
        [(y, "11", 0.1, 5.0, 1.0) for y in (2013, 2014, 2016)],
        "year int, sector string, trade_shock double, trade_pct_gdp double, net_exposure_pct_gdp double",
    )
    fx = spark.createDataFrame([(2013, 1.03), (2014, 1.10), (2016, 1.33)], "year int, cad_per_usd double")
    growth = {r.year: r.revenue_growth for r in metrics.firm_panel(revenue, exposure, fx).collect()}
    assert growth[2013] is None
    assert math.isclose(growth[2014], math.log(110 / 100))
    assert growth[2016] is None  # a 2-year change isn't labelled as a 1-year one
