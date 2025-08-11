# DataNodes — Authoring Guide

> This guide defines how to implement DataNodes in MainSequence. It’s concise, opinionated, and matches the current behavior of the base classes (which already **sanitize columns** and **validate index schema** automatically).

---

## What is a DataNode?

A **DataNode** is a unit of computation in the MainSequence DAG. Each node:

- Declares its **dependencies()** on other nodes.
- Computes and returns a **pandas.DataFrame** in **update()**.
- Optionally exposes **metadata** for discovery and documentation.
- Follows strict index/column rules so downstream nodes, storage, and tooling stay consistent.

---

## MUST vs SHOULD

### MUST (hard requirements)

- Implement `dependencies()` and `update()`.  
  If there are no dependencies, **return `{}`** (do not `pass`).
- Return a DataFrame indexed by time:
  - **Either** a `pandas.DatetimeIndex` named **`time_index`** (timezone-aware **UTC**),  
  - **Or** a `pandas.MultiIndex` where:
    - Level 0 is named **`time_index`** (timezone-aware **UTC**), and
    - Level 1 is named **`unique_identifier`**,
    - Additional levels are optional but **must be named**.
- **No datetime columns** (dates only in the index, never as DataFrame columns).
- Columns must be **lowercase**, **≤ 63 characters**, and stable (avoid renaming).
- Keep behavior deterministic for hashing/versioning:
  - `storage_hash` is derived from `__init__` args (minus ignored fields).
  - Use `_ARGS_IGNORE_IN_STORAGE_HASH` for fields that **should not** affect the storage hash.
> **Note:** The base `DataNode` class will automatically sanitize column names and validate your index schema after `update()` returns.

---

### SHOULD (best practices)

- Clear docstrings explaining inputs, outputs, and assumptions.
- Keep `update()` pure/idempotent for a given input window.
- Log only high-level progress; avoid noisy per-row logging.
- Use small helper functions for repeated IO/transform logic.
- Provide `get_table_metadata()` and `get_column_metadata()` when applicable.
- Avoid hard-coding constants; pass configuration via `__init__`.

---

---

## Canonical Template

```python
from typing import Dict, Union, List, Optional
import datetime, pytz, pandas as pd
import mainsequence.client as ms_client
from mainsequence.tdag import DataNode, APIDataNode
from mainsequence.client.models_tdag import UpdateStatistics, ColumnMetaData

UTC = pytz.utc

class MyDataNode(DataNode):
    """Example DataNode.

    Returns:
        DataFrame indexed by ("time_index","unique_identifier", ...)
        or single DatetimeIndex, with lowercase columns (≤63 chars) and no datetime columns.
    """

    _ARGS_IGNORE_IN_STORAGE_HASH = ["asset_list"]  # adjust as needed

    def __init__(self, asset_list: Optional[List[ms_client.Asset]] = None, *args, **kwargs):
        self.asset_list = asset_list or []
        super().__init__(*args, **kwargs)

    def dependencies(self) -> Dict[str, Union["DataNode", "APIDataNode"]]:
        # Return other DataNodes this node depends on
        return {}  # e.g., {"prices": self.prices_ts}

    def get_asset_list(self) -> List[ms_client.Asset]:
        """Assets this DataNode processes (used by some storage/update flows)."""
        return self.asset_list

    def get_column_metadata(self) -> List[ColumnMetaData]:
        """Describe columns for UI & documentation."""
        return [ColumnMetaData(
            column_name="my_value",
            dtype="float",
            label="My Value",
            description="Example metric"
        )]

    def get_table_metadata(self) -> Optional[ms_client.TableMetaData]:
        """Describe the table for storage."""
        return ms_client.TableMetaData(
            identifier="my_timeseries_identifier",
            data_frequency_id=ms_client.DataFrequency.one_d,
            description="Explain what this series represents"
        )

    def update(self) -> pd.DataFrame:
        """Build and return the output DataFrame."""
        us: UpdateStatistics = self.update_statistics

        # Determine update window
        start = (us.max_time_index_value or datetime.datetime(2024, 1, 1, tzinfo=UTC)).replace(tzinfo=UTC)
        start = (start + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = datetime.datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0) - datetime.timedelta(days=1)
        if start > end:
            return pd.DataFrame()

        # Build rows for each asset
        idx = pd.date_range(start=start, end=end, freq="D", tz=UTC)
        rows = []
        for asset in us.asset_list:
            df = pd.DataFrame({"my_value": 0.0}, index=idx)
            df.index.name = "time_index"
            df["unique_identifier"] = asset.unique_identifier
            df = df.set_index("unique_identifier", append=True)
            rows.append(df)

        return pd.concat(rows) if rows else pd.DataFrame()
```

### Optional: Single-Index example (one series, no per-asset level)

```python
class SingleIndexExample(DataNode):
    """Returns a single time series with a DatetimeIndex named 'time_index'."""

    def dependencies(self) -> Dict[str, Union["DataNode", "APIDataNode"]]:
        return {}

    def update(self) -> pd.DataFrame:
        us: UpdateStatistics = self.update_statistics

        start = (us.max_time_index_value or datetime.datetime(2024, 1, 1, tzinfo=UTC)).replace(tzinfo=UTC)
        start = (start + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = datetime.datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0) - datetime.timedelta(days=1)
        if start > end:
            return pd.DataFrame()

        idx = pd.date_range(start=start, end=end, freq="D", tz=UTC, name="time_index")
        return pd.DataFrame({"my_value": 0.0}, index=idx)



```

## Common Pitfalls
* Wrong index order → Always put time_index first when using MultiIndex.
* Naive datetimes → Must be timezone-aware UTC.
* Datetime columns → Forbidden; all time info belongs in the index.
* dependencies() returning pass → Must return {} if no dependencies.
* Long column names → Keep ≤ 63 chars.
* Non-deterministic update logic → Avoid random outputs unless seeded with fixed values.