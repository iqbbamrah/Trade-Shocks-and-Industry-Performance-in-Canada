"""Regressions for Q3 and Q5: Spark ML for the fit, statsmodels for the inference.

Each model is a Spark ML Pipeline: StringIndexer -> OneHotEncoder turn the
sector and year fixed effects into dummy columns, VectorAssembler packs
everything into the single `features` vector Spark ML models take, then
LinearRegression. `solver="normal"` solves OLS exactly (instead of
iteratively), which is what makes standard errors available at all.

Those standard errors are classical OLS ones, which assume every row is an
independent observation. Here they aren't: trade_shock is measured per
sector x year, so every 4-digit industry in a sector-year shares one value
(6,562 rows, but only 198 distinct shocks), and classical standard errors
come out too small. The fix is standard errors clustered by sector, which
Spark ML doesn't provide. The modeling table is small once Spark has built
it, so it's collected to pandas and refit with statsmodels for clustered
inference. The two fits must agree on the coefficients, which is checked.
With only ~18 sector clusters, the clustered p-values use a t distribution
with (clusters - 1) degrees of freedom and should still be read cautiously.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import statsmodels.formula.api as smf
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


@dataclass
class OlsResult:
    name: str
    formula: str
    n_observations: int
    n_clusters: int
    r2: float
    # term, estimate, std_error, p_value (classical, Spark ML),
    # cluster_std_error, cluster_p_value (clustered by sector, statsmodels; key terms only)
    coefficients: pd.DataFrame


def _spark_ols(data: DataFrame, label: str, numeric: list[str], fixed_effects: list[str]):
    indexed = [f"{c}_idx" for c in fixed_effects]
    dummies = [f"{c}_fe" for c in fixed_effects]
    pipeline = Pipeline(stages=[
        # alphabetAsc makes the dropped reference category deterministic.
        StringIndexer(inputCols=fixed_effects, outputCols=indexed, stringOrderType="alphabetAsc"),
        # dropLast=True drops one level per effect, avoiding the dummy-variable trap.
        OneHotEncoder(inputCols=indexed, outputCols=dummies, dropLast=True),
        VectorAssembler(inputCols=numeric + dummies, outputCol="features"),
        LinearRegression(featuresCol="features", labelCol=label, solver="normal"),
    ])
    model = pipeline.fit(data)
    lr = model.stages[-1]

    # Recover a readable name for each position in the features vector from
    # the metadata VectorAssembler attaches to it. The intercept comes last
    # in Spark's standard-error / p-value arrays.
    attrs = model.transform(data.limit(1)).schema["features"].metadata["ml_attr"]["attrs"]
    positions = sorted((a["idx"], a["name"]) for group in attrs.values() for a in group)
    terms = [name for _, name in positions] + ["intercept"]
    table = pd.DataFrame({
        "term": terms,
        "estimate": list(lr.coefficients) + [lr.intercept],
        "std_error": lr.summary.coefficientStandardErrors,
        "p_value": lr.summary.pValues,
    })
    return table, lr.summary


def fit_ols(
    df: DataFrame,
    name: str,
    label: str,
    numeric: list[str],
    fixed_effects: list[str],
    cluster_by: str = "sector",
) -> OlsResult:
    # Casting fixed effects to string makes them categories, not numbers with a slope.
    categorical = list(dict.fromkeys([*fixed_effects, cluster_by]))
    data = df.select(label, *numeric, *[F.col(c).cast("string").alias(c) for c in categorical]).dropna()

    spark_table, summary = _spark_ols(data, label, numeric, fixed_effects)

    formula = f"{label} ~ {' + '.join(numeric)} + " + " + ".join(f"C({c})" for c in fixed_effects)
    pdf = data.toPandas()  # the modeling table is small; collecting it is safe
    clustered = smf.ols(formula, data=pdf).fit(
        cov_type="cluster", cov_kwds={"groups": pdf[cluster_by]}, use_t=True
    )

    # Same model, two libraries: the key coefficients must match. (Intercepts
    # differ because each library drops a different reference category.)
    for term in numeric:
        spark_est = spark_table.loc[spark_table.term == term, "estimate"].item()
        if abs(spark_est - clustered.params[term]) > 1e-6 * max(1.0, abs(spark_est)):
            raise AssertionError(f"{name}: Spark ML and statsmodels disagree on {term}")

    key = spark_table.term.isin(numeric)
    spark_table.loc[key, "cluster_std_error"] = spark_table.loc[key, "term"].map(clustered.bse)
    spark_table.loc[key, "cluster_p_value"] = spark_table.loc[key, "term"].map(clustered.pvalues)
    return OlsResult(name, formula, summary.numInstances, pdf[cluster_by].nunique(), summary.r2, spark_table)


def fit_models(panel: DataFrame) -> list[OlsResult]:
    return [
        # Q3: are trade shocks and exposure associated with revenue levels,
        # once firm size and sector and year effects are controlled for?
        # (Size cohort alone explains ~68% of log revenue variation.)
        fit_ols(
            panel, "q3_trade_shocks_revenue",
            label="log_revenue",
            numeric=["trade_shock", "net_exposure_pct_gdp", "is_small"],
            fixed_effects=["sector", "year"],
        ),
        # Q5: do small firms' revenue growth respond differently to trade shocks?
        fit_ols(
            panel, "q5_firm_size_shock_response",
            label="revenue_growth",
            numeric=["trade_shock", "is_small", "shock_x_small"],
            fixed_effects=["sector", "year"],
        ),
    ]
