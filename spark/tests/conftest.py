import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trade_pipeline.session import get_spark  # noqa: E402


@pytest.fixture(scope="session")
def spark():
    # Test inputs are a handful of rows: two cores and one shuffle partition
    # avoid scheduling dozens of empty tasks per query.
    os.environ.setdefault("SPARK_MASTER", "local[2]")
    session = get_spark("trade-pipeline-tests")
    session.conf.set("spark.sql.shuffle.partitions", "1")
    yield session
    session.stop()
