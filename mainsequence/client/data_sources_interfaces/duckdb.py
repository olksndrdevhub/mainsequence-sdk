from __future__ import annotations

import os
from typing import Optional, Literal, List, Dict

import duckdb, pandas as pd
from pathlib import Path
import datetime
from mainsequence.logconf import logger
import pyarrow as pa
import pyarrow.parquet as pq
from ..utils import DataFrequency
import uuid
from pyarrow import fs

def get_logger():
    global logger

    # If the logger doesn't have any handlers, create it using the custom function
    logger.bind(sub_application="duck_db_interface")
    return logger

logger = get_logger()

class DuckDBInterface:
    """
    Persist/serve (time_index, unique_identifier, …) DataFrames in a DuckDB file.
    """

    def __init__(self, db_path: Optional[str | Path] = None,
                 ):
        """
        Initializes the interface with the path to the DuckDB database file.

        Args:
            db_path (Optional[str | Path]): Path to the database file.
                                             Defaults to the value of the DUCKDB_PATH
                                             environment variable or 'analytics.duckdb'
                                             in the current directory if the variable is not set.
        """
        from mainsequence.tdag.config import TDAG_DATA_PATH
        logger.warning("PASS THE RIGHT FREQUENCY")
        # ── choose default & normalise to string ───────────────────────────
        default_path = os.getenv(
            "DUCKDB_PATH",
            os.path.join(f"{TDAG_DATA_PATH}", "duck_db"),
        )
        db_uri = str(db_path or default_path).rstrip("/")

        # ── FileSystem abstraction (works for local & S3) ──────────────────
        self._fs, self._object_path = fs.FileSystem.from_uri(db_uri)

        # ── DuckDB connection ──────────────────────────────────────────────
        #   • local   → store meta‑data in a .duckdb file under db_uri
        #   • remote  → in‑memory DB; still works because all user data
        #               lives in Parquet on the object store
        if db_uri.startswith("s3://") or db_uri.startswith("gs://"):
            self.con = duckdb.connect(":memory:")
            # duckdb needs the httpfs extension for S3
            self.con.execute("INSTALL httpfs;")
            self.con.execute("LOAD httpfs;")
        else:
            meta_file = Path(db_uri) / "duck_meta.duckdb"
            meta_file.parent.mkdir(parents=True, exist_ok=True)
            self.con = duckdb.connect(str(meta_file))

        # ── sane defaults ──────────────────────────────────────────────────
        self.con.execute("PRAGMA threads = 4")
        self.con.execute("PRAGMA enable_object_cache = true")
        self.con.execute("SET TIMEZONE = 'UTC';")

        self.db_path = db_uri  # keep the fully‑qualified URI

    def launch_gui(self, host='localhost', port=4213, timeout=0.5):
        import duckdb
        import socket

        def ui_is_running(host, port, timeout):
            """Returns True if something is listening on host:port."""
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                try:
                    s.connect((host, port))
                    return True
                except (ConnectionRefusedError, socket.timeout):
                    return False

        # 1. Connect to your database
        conn = duckdb.connect(self.db_path)

        # 2. Decide whether to start the UI
        url = f"http://{host}:{port}"
        if not ui_is_running(host, port, timeout):
            # (first‐time only) install and load the UI extension
            # conn.execute("INSTALL ui;")
            # conn.execute("LOAD ui;")
            # spin up the HTTP server and open your browser
            conn.execute("CALL start_ui();")
            print(f"DuckDB Explorer launched at {url}")
        else:
            print(f"DuckDB Explorer is already running at {url}")

    # ──────────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────────
    def upsert(self, df: pd.DataFrame, table: str,
               data_frequency:DataFrequency=DataFrequency.one_m
               ) -> None:
        """
        Idempotently writes a DataFrame into *table* using (time_index, uid) PK.
        Extra columns are added to the table automatically.

        Args:
            df (pd.DataFrame): DataFrame to upsert.
            table (str): Target table name.
        """
        if df.empty:
            logger.warning(f"Attempted to upsert an empty DataFrame to table '{table}'. Skipping.")
            return

        # —— basic hygiene ——--------------------------------------------------
        df = df.copy()
        df["time_index"] = pd.to_datetime(df["time_index"], utc=True)
        if "unique_identifier" not in df.columns:
            df["unique_identifier"] = ""  # degenerate PK for daily data

        # —— derive partition columns ——---------------------------------------
        partitions = self._partition_keys(df["time_index"],data_frequency=data_frequency)
        for col, values in partitions.items():
            df[col] = values
        part_cols = list(partitions)

        logger.debug(f"Starting upsert of {len(df)} rows into table '{table}' in {self.db_path}")

        # —— de‑duplication inside *this* DataFrame ——--------------------------
        df = (
            df.sort_values(["unique_identifier", "time_index"])
            .drop_duplicates(subset=["time_index", "unique_identifier"],
                             keep="last")
        )

        # ──  Write each partition safely ─────────────────────────────────
        for keys, sub in df.groupby(part_cols, sort=False):
            part_path = self._partition_path(dict(zip(part_cols, keys)), table=table)
            self._fs.create_dir(part_path, recursive=True)

            # 4a) Cross-file de-dup: drop any rows already on disk
            try:
                existing = (
                    self.con
                    .execute(
                        f"SELECT time_index, unique_identifier "
                        f"FROM parquet_scan('{part_path}/*.parquet', hive_partitioning=TRUE)"
                    )
                    .fetch_df()
                )
            except Exception:
                existing = pd.DataFrame(columns=["time_index", "unique_identifier"])

            if not existing.empty:
                merged = sub.merge(
                    existing,
                    on=["time_index", "unique_identifier"],
                    how="left",
                    indicator=True
                )
                sub = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
            if sub.empty:
                continue  # nothing new here

            # 4b) Atomic write: tmp → final
            ts = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
            final_name = f"part-{ts}-{uuid.uuid4().hex}.parquet"
            tmp_name = final_name + ".tmp"

            tmp_path = f"{part_path}/{tmp_name}"
            final_path = f"{part_path}/{final_name}"

            # Sort for zone-maps, convert to Arrow
            sub = sub.sort_values(["unique_identifier", "time_index"])
            table_arrow = pa.Table.from_pandas(sub, preserve_index=False)

            # Write to *.tmp
            pq.write_table(
                table_arrow,
                where=tmp_path,
                row_group_size=512_000,
                compression="zstd",
                filesystem=self._fs,

                version="2.6",  # enable the Parquet logical type TIMESTAMP_NANOS
                coerce_timestamps=None,  # don’t downcast; let 2.6 write native ns
                allow_truncated_timestamps=False,  # fail if you ever would lose precision

                # (optional) for maximum compatibility via INT96:
                # use_deprecated_int96_timestamps=True,
            )
            # Rename into place (tmp → .parquet)
            self._fs.move(tmp_path, final_path)

        # ──  Refresh view ────────────────────────────────────────────────
        self._ensure_view(df,table=table)


    def read(
            self,
            table: str,data_frequency:DataFrequency=DataFrequency.one_m,
            *,
            start: Optional[datetime.datetime] = None,
            end: Optional[datetime.datetime] = None,
            great_or_equal: bool = True,  # Changed back to boolean
            less_or_equal: bool = True,  # Changed back to boolean
            ids: Optional[List[str]] = None,
            columns: Optional[List[str]] = None,
            unique_identifier_range_map: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Reads data from the specified table, with optional filtering.
        Handles missing tables by returning an empty DataFrame.

        Args:
            table (str): The name of the table to read from.
            start (Optional[datetime.datetime]): Minimum time_index filter.
            end (Optional[datetime.datetime]): Maximum time_index filter.
            great_or_equal (bool): If True, use >= for start date comparison. Defaults to True.
            less_or_equal (bool): If True, use <= for end date comparison. Defaults to True.
            ids (Optional[List[str]]): List of specific unique_identifiers to include.
            columns (Optional[List[str]]): Specific columns to select. Reads all if None.
            unique_identifier_range_map (Optional[UniqueIdentifierRangeMap]):
                A map where keys are unique_identifiers and values are dicts specifying
                date ranges (start_date, end_date, start_date_operand, end_date_operand)
                for that identifier. Mutually exclusive with 'ids'.

        Returns:
            pd.DataFrame: The queried data, or an empty DataFrame if the table doesn't exist.

        Raises:
            ValueError: If both `ids` and `unique_identifier_range_map` are provided.
        """
        # Map boolean flags to operator strings internally
        start_operator = '>=' if great_or_equal else '>'
        end_operator = '<=' if less_or_equal else '<'

        if ids is not None and unique_identifier_range_map is not None:
            raise ValueError("Cannot provide both 'ids' and 'unique_identifier_range_map'.")

        logger.debug(
            f"Duck DB: Reading from table '{table}' with filters: start={start}, end={end}, "
            f"ids={ids is not None}, columns={columns}, range_map={unique_identifier_range_map is not None}"
        )

        cols_select = "*"
        if columns:
            required_cols = {"time_index", "unique_identifier"}
            select_set = set(columns) | required_cols
            cols_select = ", ".join(f'"{c}"' for c in select_set)

        sql_parts = [f'SELECT {cols_select} FROM "{table}"']
        params = []
        where_clauses = []

        # --- Build WHERE clauses ---
        if start is not None:
            where_clauses.append(f"time_index {start_operator} ?")
            params.append(start.replace(tzinfo=None) if start.tzinfo else start)
        if end is not None:
            where_clauses.append(f"time_index {end_operator} ?")
            params.append(end.replace(tzinfo=None) if end.tzinfo else end)
        if ids:
            if not isinstance(ids, list): ids = list(ids)
            if ids:
                placeholders = ", ".join("?" for _ in ids)
                where_clauses.append(f"unique_identifier IN ({placeholders})")
                params.extend(ids)
        if unique_identifier_range_map:
            range_conditions = []
            for uid, date_info in unique_identifier_range_map.items():
                uid_conditions = ["unique_identifier = ?"]
                range_params = [uid]
                # Use operands from map if present, otherwise default to >= and <=
                s_op = date_info.get('start_date_operand', '>=')
                e_op = date_info.get('end_date_operand', '<=')
                if date_info.get('start_date'):
                    uid_conditions.append(f"time_index {s_op} ?")
                    s_date = date_info['start_date']
                    range_params.append(s_date.replace(tzinfo=None) if s_date.tzinfo else s_date)
                if date_info.get('end_date'):
                    uid_conditions.append(f"time_index {e_op} ?")
                    e_date = date_info['end_date']
                    range_params.append(e_date.replace(tzinfo=None) if e_date.tzinfo else e_date)
                range_conditions.append(f"({' AND '.join(uid_conditions)})")
                params.extend(range_params)
            if range_conditions:
                where_clauses.append(f"({' OR '.join(range_conditions)})")

        if where_clauses: sql_parts.append("WHERE " + " AND ".join(where_clauses))
        sql_parts.append("ORDER BY time_index")
        query = " ".join(sql_parts)
        logger.debug(f"Executing read query: {query} with params: {params}")

        try:
            table_exists_result = self.con.execute("""
                                                            SELECT COUNT(*) 
                                                              FROM information_schema.tables
                                                             WHERE table_schema='main' AND table_name = ?
                                                            UNION ALL
                                                            SELECT COUNT(*) 
                                                              FROM information_schema.views
                                                             WHERE table_schema='main' AND table_name = ?
                                                        """, [table, table]).fetchone()[0] > 0

            if table_exists_result is None:
                logger.warning(f"Table '{table}' does not exist in {self.db_path}. Returning empty DataFrame.")
                return pd.DataFrame()

            df = self.con.execute(query, params).fetch_df()

            if not df.empty:
                schema = self.con.execute(f'PRAGMA table_info("{table}")').fetchall()
                type_map = {
                    name: self._duck_to_pandas(duck_type, data_frequency=data_frequency)
                    for cid, name, duck_type, notnull, default, pk in schema
                    if name in df.columns
                }
                for col, target_type in type_map.items():
                    try:
                        if target_type == "datetime64[ns, UTC]":
                            arr =df[col].values
                            arr_ns = arr.astype("datetime64[ns]")
                            df[col] =pd.Series(
                                    pd.DatetimeIndex(arr_ns, tz="UTC"),
                                    index=df.index,
                                    name=col,
                                )
                        elif target_type == "datetime64[ns]":
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                        else:
                            if isinstance(target_type, (pd.Int64Dtype, pd.BooleanDtype, pd.StringDtype)):
                                df[col] = df[col].astype(target_type, errors='ignore')
                            else:
                                df[col] = df[col].astype(target_type, errors='ignore')
                    except Exception as type_e:
                        logger.warning(f"Could not coerce column '{col}' to type '{target_type}': {type_e}")

                logger.debug(f"Read {len(df)} rows from table '{table}'.")
                return df

            return pd.DataFrame()

        except duckdb.CatalogException as e:
            logger.warning(f"CatalogException for table '{table}': {e}. Returning empty DataFrame.")
            return pd.DataFrame()
        except duckdb.Error as e:
            logger.error(f"Failed to read data from table '{table}': {e}")
            raise
        except Exception as e:
            logger.exception(f"An unexpected error occurred during read from table '{table}': {e}")
            raise

    def drop_table(self, table: str) -> None:
        """
        Drops the specified table and corresponding view from the database.

        Args:
            table (str): The name of the table/view to drop.
        """
        logger.debug(f"Attempting to drop table and view '{table}' from {self.db_path}")
        try:
            # Drop the view first (if it exists)
            self.con.execute(f'DROP VIEW IF EXISTS "{table}"')
            logger.debug(f"Dropped view '{table}' (if it existed).")

            # Then drop the table (if it exists)
            self.con.execute(f'DROP TABLE IF EXISTS "{table}"')
            logger.debug(f"Dropped table '{table}' (if it existed).")

        except duckdb.Error as e:
            logger.error(f"Failed to drop table/view '{table}': {e}")
            raise
        except Exception as e:
            logger.exception(f"An unexpected error occurred while dropping table/view '{table}': {e}")
            raise

    def list_tables(self) -> List[str]:
        """
        Returns names of all tables and views in the main schema.
        """
        try:
            rows = self.con.execute("SHOW TABLES").fetchall()
            return [r[0] for r in rows]
        except duckdb.Error as e:
            logger.error(f"Error listing tables/views in {self.db_path}: {e}")
            return []


    # ──────────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────────

    def _ensure_view(self, df: pd.DataFrame, table: str) -> None:
        """
        CREATE OR REPLACE a view named `table` that:
          * reads all Parquet under self.db_path/table/**
          * hides partition columns (year, month, day)
          * locks column dtypes by explicit CASTs
        """
        partition_cols = {"year", "month", "day"}
        select_exprs = []
        for col, dtype in df.dtypes.items():
            if col in partition_cols:
                continue
            duck_type = self._pandas_to_duck(dtype)
            select_exprs.append(f"CAST({col} AS {duck_type}) AS {col}")

        star = ",\n       ".join(select_exprs) or "*"
        file_glob = f"{self.db_path}/{table}/**/*.parquet"
        ddl = f"""
        CREATE OR REPLACE VIEW "{table}" AS
        SELECT {star}
        FROM parquet_scan('{file_glob}', hive_partitioning=TRUE)
        """

        # Execute DDL in an explicit transaction
        self._execute_transaction(ddl)

    def _partition_path(self, keys: dict,table:str) -> str:
        parts = [f"{k}={int(v):02d}" if k != "year" else f"{k}={int(v):04d}"
                 for k, v in keys.items()]
        return f"{self.db_path}/{table}/" + "/".join(parts)

    def _partition_keys(self, ts: pd.Series,data_frequency:DataFrequency) -> dict:
        """Return a dict of partition column → Series."""
        keys = {"year": ts.dt.year, "month": ts.dt.month}
        if data_frequency == "minute":
            keys["day"] = ts.dt.day
        return keys

    def _execute_transaction(self, sql: str) -> None:
        """
        Run a single-statement SQL in a BEGIN/COMMIT block,
        rolling back on any failure.
        """
        try:
            self.con.execute("BEGIN TRANSACTION;")
            self.con.execute(sql)
            self.con.execute("COMMIT;")
        except Exception:
            # best-effort rollback (if inside a failed transaction)
            try:
                self.con.execute("ROLLBACK;")
            except Exception:
                pass
            raise
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
    def _duck_to_pandas(duck_type: str,data_frequency:DataFrequency):
        """
        Minimal DuckDB → pandas dtype mapping.
        Returns the dtype object (preferred) so that
        `df.astype({...})` gets pandas’ nullable dtypes.
        Extend as needed.
        """
        dt = duck_type.upper()

        # --- datetimes ------------------------------------------------------
        if dt in ("TIMESTAMPTZ", "TIMESTAMP WITH TIME ZONE"):
            # keep the UTC tz-awareness
            return "datetime64[ns, UTC]"


        if dt in ("TIMESTAMP", "DATETIME",):
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

        # ─────────────────────────────────────────────────────────────────────── #
        # 3. OVERNIGHT DEDUP & COMPACTION                                        #
        # ─────────────────────────────────────────────────────────────────────── #
        def overnight_dedup(self, table: str, date: Optional[datetime.date] = None) -> None:
            """
            Keep only the newest row per (time_index, unique_identifier)
            for each partition, coalesce small files into one Parquet file.

            Run this once a day during low‑traffic hours.
            """
            # --- select partitions to touch ------------------------------------
            base = f"{self.db_path}/{table}"
            selector = fs.FileSelector(base, recursive=True)
            dirs = {info.path.rpartition("/")[0] for info in self._fs.get_file_info(selector)
                    if info.type == fs.FileType.File
                    and info.path.endswith(".parquet")}

            if date:
                y, m, d = date.year, date.month, date.day
                dirs = {p for p in dirs if
                        f"year={y:04d}" in p and f"month={m:02d}" in p
                        and (data_frequency != "minute" or f"day={d:02d}" in p)}

            for part_path in sorted(dirs):
                tmp_file = f"{part_path}/compact-{uuid.uuid4().hex}.parquet"

                # DuckDB SQL: window‑deduplicate & write in one shot
                copy_sql = f"""
                COPY (
                  SELECT *
                  FROM (
                    SELECT *,
                           ROW_NUMBER() OVER (
                               PARTITION BY time_index, unique_identifier
                               ORDER BY _file_path DESC
                           ) AS rn
                    FROM parquet_scan('{part_path}/*.parquet',
                                      hive_partitioning=TRUE,
                                      filename=true)       -- exposes _file_path
                  )
                  WHERE rn = 1
                )
                TO '{tmp_file}'
                (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 512000)
                """
                self.con.execute(copy_sql)

                # remove old fragments & leave only the compacted file
                for info in self._fs.get_file_info(fs.FileSelector(part_path)):
                    if info.type == fs.FileType.File and info.path != tmp_file:
                        self._fs.delete_file(info.path)

                # Optionally rename to a deterministic name; here we just keep tmp_file
                logger.info(f"Compacted + de‑duplicated partition {part_path}")