"""Parquet reads and writes for every layer.

Spark's own Parquet reader/writer goes through Hadoop's local filesystem, which
on Windows needs winutils.exe and hadoop.dll (found via HADOOP_HOME). Apache
doesn't publish those binaries, so on Windows without them this module falls
back to Apache Arrow for the file I/O only. The on-disk layout is identical
(hive-style `column=value/` partition folders), and every transformation still
runs in Spark. On Linux, WSL, Docker, or Windows with HADOOP_HOME set, Spark's
native reader/writer is used.
"""
from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path
from typing import Sequence

import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
from pyspark.sql import DataFrame, SparkSession


def native_parquet_supported() -> bool:
    return os.name != "nt" or bool(os.environ.get("HADOOP_HOME"))


def write_parquet(df: DataFrame, path: Path, partition_by: Sequence[str] = ()) -> None:
    if native_parquet_supported():
        writer = df.write.mode("overwrite")
        if partition_by:
            writer = writer.partitionBy(*partition_by)
        writer.parquet(str(path))
        return

    shutil.rmtree(path, ignore_errors=True)
    table = df.toArrow()
    if partition_by:
        pq.write_to_dataset(table, str(path), partition_cols=list(partition_by))
    else:
        path.mkdir(parents=True)
        pq.write_table(table, str(path / "part-00000.parquet"))


def read_parquet(spark: SparkSession, path: Path) -> DataFrame:
    if native_parquet_supported():
        return spark.read.parquet(str(path))

    table = pads.dataset(str(path), format="parquet", partitioning="hive").to_table()
    # Partition columns can come back dictionary-encoded; Spark wants plain types.
    plain_schema = pa.schema(
        pa.field(f.name, f.type.value_type if pa.types.is_dictionary(f.type) else f.type)
        for f in table.schema
    )
    return spark.createDataFrame(table.cast(plain_schema))


def ensure_unzipped(csv_path: Path) -> Path:
    """Spark can't read .zip archives, so extract `X.csv` from `X.csv.zip` once."""
    if not csv_path.exists():
        with zipfile.ZipFile(csv_path.with_name(csv_path.name + ".zip")) as archive:
            archive.extract(csv_path.name, csv_path.parent)
    return csv_path
