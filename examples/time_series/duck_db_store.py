from __future__ import annotations
import duckdb, pandas as pd
from pathlib import Path
import datetime
import pytz
class TimeSeriesStore:
    """
    Persist/serve (time_index, unique_identifier, …) DataFrames in a DuckDB file.
    """

    def __init__(self, db_path: str | Path = "analytics.duckdb"):
        self.db_path = Path(db_path)
        # Single connection kept open – DuckDB is thread‑safe for concurrent readers
        self.con = duckdb.connect(str(self.db_path))
        # Sane defaults
        self.con.execute("PRAGMA threads = 4")  # tweak to your cores
        self.con.execute("PRAGMA enable_object_cache = true")

    # ──────────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────────
    def upsert(self, df: pd.DataFrame, table: str) -> None:
        """
        idempotently writes a DataFrame into *table* using (time_index, uid) PK.
        Extra columns are added to the table automatically.
        """
        df = df.copy()  # keep caller's frame untouched

        # 1) Ensure mandatory columns exist
        if not {"time_index", "unique_identifier"} <= set(df.columns):
            raise ValueError("DataFrame must contain time_index and unique_identifier")

        # 2) Create/extend the table as needed
        self._ensure_table(table, df)

        # 3) Register the frame as an in‑memory DuckDB relation
        self.con.register("upd_df", df)

        # 4) Upsert – INSERT OR REPLACE uses the PRIMARY KEY to deduplicate
        cols = ", ".join(df.columns)
        self.con.execute(f"""
            INSERT OR REPLACE INTO "{table}" ({cols})
            SELECT {cols} FROM upd_df
        """)
        self.con.unregister("upd_df")

    def read(
        self,
        table: str,
        *,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        ids: list[str] | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Return a slice as a pandas DataFrame, fast.
        """
        cols = ", ".join(columns) if columns else "*"
        sql = [f'SELECT {cols} FROM "{table}"']
        params: list = []

        if start is not None:
            sql.append("time_index >= ?")
            params.append(start)
        if end is not None:
            sql.append("time_index <= ?")
            params.append(end)
        if ids:
            placeholders = ",".join("?" for _ in ids)
            sql.append(f"unique_identifier IN ({placeholders})")
            params.extend(ids)

        if len(sql) > 1:
            sql = [sql[0], "WHERE", " AND ".join(sql[1:])]
        sql.append("ORDER BY time_index")

        return self.con.execute(" ".join(sql), params).fetch_df()

    # ──────────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────────
    def _ensure_table(self, table: str, df: pd.DataFrame) -> None:
        """
        Creates the table if absent, or adds any still‑missing columns.
        """
        # 1) Create‑if‑missing with the two key columns
        self.con.execute(f"""
           CREATE TABLE IF NOT EXISTS "{table}" (
    time_index        TIMESTAMPTZ   NOT NULL,
                unique_identifier VARCHAR   NOT NULL,
                PRIMARY KEY (time_index, unique_identifier)
            )
        """)

        # 2) Add new columns on the fly
        existing_cols = {
            r[1] for r in self.con.execute(f"""
                PRAGMA table_info('{table}')
            """).fetchall()
        }
        for col, dtype in df.dtypes.items():
            if col in existing_cols:
                continue
            duck_type = self._pandas_to_duck(dtype)
            self.con.execute(f'ALTER TABLE "{table}" ADD COLUMN {col} {duck_type}')

    @staticmethod
    def _pandas_to_duck(dtype) -> str:
        """
        Minimal dtype → DuckDB mapping. Extend as needed.
        """
        if (pd.api.types.is_datetime64_any_dtype(dtype)
                or pd.api.types.is_datetime64tz_dtype(dtype)):
            return "TIMESTAMPTZ"
        if pd.api.types.is_integer_dtype(dtype):
            return "BIGINT"
        if pd.api.types.is_float_dtype(dtype):
            return "DOUBLE"
        if pd.api.types.is_bool_dtype(dtype):
            return "BOOLEAN"
        return "VARCHAR"



def your_function() -> tuple[pd.DataFrame, str]:
    """
    Demo producer that manufactures a DataFrame and a target‑table name.

    Returns
    -------
    df : pandas.DataFrame
        Columns: time_index (UTC timestamp), unique_identifier (str), value (float)
    table_name : str
        Where you want to store the frame in DuckDB
    """
    import datetime
    import numpy as np
    now = pd.Timestamp.utcnow().floor("s")
    df = pd.DataFrame({
        "time_index": [now + datetime.timedelta(minutes=5 * i) for i in range(12)],
        "unique_identifier": ["A", "B", "C"] * 4,
        "value": [round(np.random.uniform(-1, 1), 3) for _ in range(12)],
    })
    return df, "sample_table"

store = TimeSeriesStore("analytics.duckdb")

# Somewhere in your pipeline
df, tbl = your_function()       # returns a DataFrame and a table name
if df["time_index"].dt.tz is None:
    df["time_index"] = df["time_index"].dt.tz_localize("UTC")
else:
    df["time_index"] = df["time_index"].dt.tz_convert("UTC")
store.upsert(df, tbl)           # idempotent write



start_dt = datetime.datetime(2025, 4, 1,  0,  0,  0,      tzinfo=pytz.utc)
end_dt   = datetime.datetime(2025, 4, 21, 23, 59, 59, 999999, tzinfo=pytz.utc)
# Fast read back
slice_df = store.read(
    tbl,
    start=start_dt,
    end=end_dt,
    ids=["ABC", "XYZ"],
    columns=["time_index", "unique_identifier", "value"],
)

print(slice_df)

duckdb.sql("CALL start_ui();")