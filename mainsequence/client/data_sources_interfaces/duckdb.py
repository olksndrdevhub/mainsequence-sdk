from __future__ import annotations
import duckdb, pandas as pd
from pathlib import Path
import datetime
import pytz
class DuckDBInterface:
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

    def read(self, table: str, *, start=None, end=None,
             ids=None, columns=None) -> pd.DataFrame:

        cols = ", ".join(columns) if columns else "*"
        sql = [f'SELECT {cols} FROM "{table}"']
        parms = []

        if start is not None:
            sql += ["WHERE" if len(sql) == 1 else "AND", "time_index >= ?"]
            parms.append(start)
        if end is not None:
            sql += ["AND", "time_index <= ?"]
            parms.append(end)
        if ids:
            placeholders = ",".join("?" for _ in ids)
            sql += ["AND", f"unique_identifier IN ({placeholders})"]
            parms.extend(ids)

        sql.append("ORDER BY time_index")

        df = self.con.execute(" ".join(sql), parms).fetch_df()

        # ▸ Grab DuckDB’s column types and coerce explicitly
        schema = self.con.execute(f"DESCRIBE {table}").fetchall()
        type_map = {name: self._duck_to_pandas(duck_type)
                    for name, duck_type, *_ in schema
                    if name in df.columns}

        return df.astype(type_map, errors="ignore")

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
    @staticmethod
    def _duck_to_pandas(duck_type: str):
        """
        Minimal DuckDB → pandas dtype mapping.
        Returns the dtype object (preferred) so that
        `df.astype({...})` gets pandas’ nullable dtypes.
        Extend as needed.
        """
        dt = duck_type.upper()

        # --- datetimes ------------------------------------------------------
        if dt in ("TIMESTAMPTZ", "TIMESTAMP", "DATETIME"):
            # keep timezone if present; duckdb returns tz‑aware objects already,
            # so no explicit 'UTC' suffix is needed here.
            return "datetime64[ns]"
        if dt == "DATE":
            return "datetime64[ns]"          # pandas treats it as midnight

        # --- integers -------------------------------------------------------
        if dt in ("TINYINT", "SMALLINT", "INTEGER", "INT", "BIGINT"):
            return pd.Int64Dtype()           # nullable 64‑bit int

        # --- floats / numerics ---------------------------------------------
        if dt in ("REAL", "FLOAT", "DOUBLE", "DECIMAL"):
            return "float64"

        # --- booleans -------------------------------------------------------
        if dt == "BOOLEAN":
            return pd.BooleanDtype()         # nullable boolean

        # --- everything else ------------------------------------------------
        return pd.StringDtype()              # pandas‘ native nullable string


