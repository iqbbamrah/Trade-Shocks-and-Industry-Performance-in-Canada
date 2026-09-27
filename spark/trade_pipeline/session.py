"""Builds the one SparkSession every stage shares."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession


def _configure_environment() -> None:
    # Spark 4 needs Java 17+. An explicit JAVA_HOME wins; otherwise use a
    # portable JDK at ~/tools/jdk-21 if one exists, else whatever `java` is on PATH.
    if "JAVA_HOME" not in os.environ:
        portable_jdk = Path.home() / "tools" / "jdk-21"
        if portable_jdk.exists():
            os.environ["JAVA_HOME"] = str(portable_jdk)

    # Run Spark's Python workers with this interpreter (the venv), not
    # whichever `python` happens to be first on PATH.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    # Point SPARK_HOME at the pip-installed pyspark so Spark skips its
    # find-spark-home script, which misparses paths containing " - ".
    if "SPARK_HOME" not in os.environ:
        import pyspark

        os.environ["SPARK_HOME"] = str(Path(pyspark.__file__).parent)


def get_spark(app_name: str = "trade-shocks-pipeline") -> SparkSession:
    _configure_environment()
    spark = (
        SparkSession.builder.appName(app_name)
        # local[*]: driver and executors in one JVM, one task slot per CPU core.
        .master(os.environ.get("SPARK_MASTER", "local[*]"))
        # The default of 200 shuffle partitions is sized for clusters; on one
        # machine with ~1M rows it just means 200 tiny tasks per shuffle.
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.session.timeZone", "UTC")
        # Arrow makes Spark <-> pandas/pyarrow conversions columnar and fast.
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark
