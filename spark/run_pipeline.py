"""Run the Trade Shocks PySpark pipeline: bronze -> silver -> gold -> models.

    python spark/run_pipeline.py                 # everything
    python spark/run_pipeline.py --from gold     # reuse existing bronze/silver output
    python spark/run_pipeline.py --explain       # also print the firm panel's query plan

While it runs, the Spark UI is at http://localhost:4040 (jobs, stages, SQL plans).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pyspark.sql import DataFrame

from trade_pipeline import checks, clean, config, ingest, metrics, models
from trade_pipeline.io import read_parquet, write_parquet
from trade_pipeline.session import get_spark

LAYERS = ["bronze", "silver", "gold", "models"]


def load(spark, layer_dir: Path, names) -> dict[str, DataFrame]:
    return {name: read_parquet(spark, layer_dir / name) for name in names}


def save(tables: dict[str, DataFrame], layer_dir: Path, partitions=None) -> dict[str, int]:
    counts = {}
    for name, df in tables.items():
        write_parquet(df, layer_dir / name, (partitions or {}).get(name, ()))
        counts[name] = read_parquet(df.sparkSession, layer_dir / name).count()
    return counts


def build_silver(spark) -> dict[str, DataFrame]:
    b = load(spark, config.BRONZE_DIR, ingest.SOURCE_FILES)
    # Revenue feeds two silver tables, so cache it rather than recompute the
    # dedupe/filter for each. The cache fills on the first action that uses it.
    revenue = clean.clean_revenue(b["revenue"]).cache()
    return {
        "revenue": revenue,
        "revenue_quartiles": clean.unpivot_revenue_quartiles(revenue),
        "expenses": clean.clean_expenses(b["expenses"]),
        "trade": clean.clean_trade(b["exports"], b["imports"]),
        "gdp": clean.clean_gdp(b["gdp"]),
        "fx": clean.clean_fx(b["fx"]),
    }


def build_gold(spark, s: dict[str, DataFrame]) -> dict[str, DataFrame]:
    growth = metrics.gdp_growth(s["gdp"])
    exposure = metrics.trade_exposure(spark, s["trade"], s["gdp"])
    panel = metrics.firm_panel(s["revenue"], exposure, s["fx"])
    return {
        "gdp_growth": growth,
        "industry_cagr": metrics.industry_cagr(growth),
        "trade_exposure": exposure,
        "firm_panel": panel,
        "fx_revenue_correlation": metrics.fx_revenue_correlation(panel),
        "revenue_distribution": metrics.revenue_distribution(s["revenue_quartiles"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="start", choices=LAYERS, default="bronze", help="first layer to (re)build")
    parser.add_argument("--explain", action="store_true", help="print the firm panel's physical plan")
    args = parser.parse_args()
    run = LAYERS[LAYERS.index(args.start):]

    spark = get_spark()
    summary: dict = {"layers": {}, "checks_passed": 0}
    start = time.perf_counter()

    def timed(layer, fn):
        t0 = time.perf_counter()
        result = fn()
        summary["layers"][layer] = {"seconds": round(time.perf_counter() - t0, 1), **(result or {})}
        print(f"[{layer}] done in {summary['layers'][layer]['seconds']}s")
        return result

    if "bronze" in run:
        timed("bronze", lambda: {"rows": save(ingest.build_bronze(spark), config.BRONZE_DIR, ingest.PARTITION_COLUMNS)})

    if "silver" in run:
        def silver():
            tables = build_silver(spark)
            summary["checks_passed"] += checks.run(checks.silver_checks(tables), "silver")
            return {"rows": save(tables, config.SILVER_DIR, {"revenue": ["year"], "revenue_quartiles": ["year"]})}
        timed("silver", silver)

    if "gold" in run:
        def gold():
            s = load(spark, config.SILVER_DIR, ["revenue", "revenue_quartiles", "trade", "gdp", "fx"])
            tables = build_gold(spark, s)
            if args.explain:
                tables["firm_panel"].explain(mode="formatted")
            summary["checks_passed"] += checks.run(checks.gold_checks(tables), "gold")
            return {"rows": save(tables, config.GOLD_DIR)}
        timed("gold", gold)

    if "models" in run:
        def fit():
            panel = read_parquet(spark, config.GOLD_DIR / "firm_panel")
            config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            results = {}
            for r in models.fit_models(panel):
                r.coefficients.to_csv(config.RESULTS_DIR / f"{r.name}.csv", index=False)
                results[r.name] = {
                    "formula": r.formula, "n": r.n_observations,
                    "sector_clusters": r.n_clusters, "r2": round(r.r2, 4),
                }
                key_terms = r.coefficients.dropna(subset=["cluster_p_value"])
                print(f"\n{r.name}: {r.formula}\n  n={r.n_observations:,}  R2={r.r2:.3f}  "
                      f"clusters={r.n_clusters}  (cluster_* = standard errors clustered by sector)")
                print(key_terms.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
            return {"models": results}
        timed("models", fit)

    summary["total_seconds"] = round(time.perf_counter() - start, 1)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUTPUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2))
    checks_note = f"; {summary['checks_passed']} data-quality checks passed" if summary["checks_passed"] else ""
    print(f"\nPipeline finished in {summary['total_seconds']}s{checks_note}.")
    spark.stop()


if __name__ == "__main__":
    main()
