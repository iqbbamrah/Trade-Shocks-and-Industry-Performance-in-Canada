"""Spark ML regressions for Q3 and Q5.

Each model is an ML Pipeline: StringIndexer -> OneHotEncoder turn the sector
and year fixed effects into dummy columns, VectorAssembler packs everything
into the single `features` vector Spark ML models take, then LinearRegression.
`solver="normal"` solves OLS exactly (instead of iteratively), which is what
makes coefficient standard errors and p-values available.

Standard errors are classical OLS ones. The original Q5 notebook used
entity/time fixed effects with clustered errors (linearmodels.PanelOLS),
which Spark ML doesn't provide, so p-values aren't directly comparable.
"""
from __future__ import annotations

from dataclasses import dataclass

from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


@dataclass
class OlsResult:
    name: str
    formula: str
    n_observations: int
    r2: float
    coefficients: DataFrame  # term, estimate, std_error, t_value, p_value


def fit_ols(
    spark: SparkSession,
    df: DataFrame,
    name: str,
    label: str,
    numeric: list[str],
    fixed_effects: list[str],
) -> OlsResult:
    data = df.select(label, *numeric, *[F.col(c).cast("string").alias(c) for c in fixed_effects]).dropna()
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
    summary = model.stages[-1].summary

    # Recover a readable name for each position in the features vector from
    # the metadata VectorAssembler attaches to it.
    attrs = model.transform(data.limit(1)).schema["features"].metadata["ml_attr"]["attrs"]
    names = [a["name"] for group in attrs.values() for a in group]
    order = [a["idx"] for group in attrs.values() for a in group]
    terms = [n for _, n in sorted(zip(order, names))] + ["intercept"]
    estimates = list(model.stages[-1].coefficients) + [model.stages[-1].intercept]

    rows = [
        (term, float(est), float(se), float(t), float(p))
        for term, est, se, t, p in zip(
            terms, estimates, summary.coefficientStandardErrors, summary.tValues, summary.pValues
        )
    ]
    coefficients = spark.createDataFrame(rows, "term string, estimate double, std_error double, t_value double, p_value double")
    formula = f"{label} ~ {' + '.join(numeric)} + " + " + ".join(f"C({c})" for c in fixed_effects)
    return OlsResult(name, formula, summary.numInstances, summary.r2, coefficients)


def fit_models(spark: SparkSession, panel: DataFrame) -> list[OlsResult]:
    return [
        # Q3: are trade shocks and exposure associated with revenue levels,
        # once sector and year effects are controlled for?
        fit_ols(
            spark, panel, "q3_trade_shocks_revenue",
            label="log_revenue",
            numeric=["trade_shock", "net_exposure_pct_gdp"],
            fixed_effects=["sector", "year"],
        ),
        # Q5: do small firms' revenue growth respond differently to trade shocks?
        fit_ols(
            spark, panel, "q5_firm_size_shock_response",
            label="revenue_growth",
            numeric=["trade_shock", "is_small", "shock_x_small"],
            fixed_effects=["sector", "year"],
        ),
    ]
