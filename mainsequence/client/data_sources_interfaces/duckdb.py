from __future__ import annotations

import os
from typing import Optional, Literal, List, Dict

import duckdb, pandas as pd
from pathlib import Path
import datetime
from mainsequence.logconf import logger

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

    def __init__(self, db_path: Optional[str | Path] = None):
        """
        Initializes the interface with the path to the DuckDB database file.

        Args:
            db_path (Optional[str | Path]): Path to the database file.
                                             Defaults to the value of the DUCKDB_PATH
                                             environment variable or 'analytics.duckdb'
                                             in the current directory if the variable is not set.
        """
        from mainsequence.tdag.config import TDAG_DATA_PATH
        default_path = Path(os.getenv("DUCKDB_PATH", os.path.join(f"{TDAG_DATA_PATH}", "duck_db")))
        self.db_path = Path(db_path) if db_path else default_path
        logger.info(f"DuckDBInterface initialized with db_path: {self.db_path}")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.con = duckdb.connect(str(self.db_path))
        # Sane defaults
        self.con.execute("PRAGMA threads = 4")  # tweak to your cores
        self.con.execute("PRAGMA enable_object_cache = true")


    # ──────────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────────
    def upsert(self, df: pd.DataFrame, table: str) -> None:
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

        df = df.copy()
        logger.info(f"Starting upsert of {len(df)} rows into table '{table}' in {self.db_path}")

        try:
            self._ensure_table(table, df)
            self.con.register("upd_df", df)
            cols = ", ".join(f'"{c}"' for c in df.columns)
            try:
                self.con.execute(f"""
                    INSERT OR REPLACE INTO "{table}" ({cols})
                    SELECT {cols} FROM upd_df
                """)
                logger.info(f"Successfully upserted {len(df)} rows into table '{table}'.")
            except duckdb.Error as e:
                logger.error(f"Error during INSERT OR REPLACE into '{table}': {e}")
                raise
            finally:
                self.con.unregister("upd_df")
                logger.debug("Unregistered temporary view 'upd_df'.")
        except duckdb.Error as e:
            logger.error(f"Failed to upsert data into table '{table}': {e}")
            raise
        except Exception as e:
            logger.exception(f"An unexpected error occurred during upsert to table '{table}': {e}")
            raise

    def read(
            self,
            table: str,
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

        logger.info(
            f"Reading from table '{table}' with filters: start={start}, end={end}, "
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
            table_exists_result = self.con.execute(
                "SELECT table_name FROM duckdb_tables WHERE table_name = ?", [table]
            ).fetchone()
            if table_exists_result is None:
                logger.warning(f"Table '{table}' does not exist in {self.db_path}. Returning empty DataFrame.")
                return pd.DataFrame()

            df = self.con.execute(query, params).fetch_df()

            if not df.empty:
                schema = self.con.execute(f'PRAGMA table_info("{table}")').fetchall()
                type_map = {name: self._duck_to_pandas(duck_type)
                            for name, duck_type, nullable, pk, default, primary_key_pos in schema
                            if name in df.columns}
                for col, target_type in type_map.items():
                    try:
                        if target_type == "datetime64[ns, UTC]":
                            df[col] = pd.to_datetime(df[col], errors='coerce').dt.tz_localize('UTC')
                        elif target_type == "datetime64[ns]":
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                        else:
                            if isinstance(target_type, (pd.Int64Dtype, pd.BooleanDtype, pd.StringDtype)):
                                df[col] = df[col].astype(target_type, errors='ignore')
                            else:
                                df[col] = df[col].astype(target_type, errors='ignore')
                    except Exception as type_e:
                        logger.warning(f"Could not coerce column '{col}' to type '{target_type}': {type_e}")

                logger.info(f"Read {len(df)} rows from table '{table}'.")
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
        Drops the specified table from the database.

        Args:
            table (str): The name of the table to drop.
        """
        logger.info(f"Attempting to drop table '{table}' from {self.db_path}")
        try:
            # Use IF EXISTS to avoid error if table is already gone
            self.con.execute(f'DROP TABLE IF EXISTS "{table}"')
            logger.info(f"Successfully dropped table '{table}' (if it existed).")
        except duckdb.Error as e:
            logger.error(f"Failed to drop table '{table}': {e}")
            raise # Re-raise the error after logging
        except Exception as e:
            logger.exception(f"An unexpected error occurred while dropping table '{table}': {e}")
            raise


    def list_tables(self) -> List[str]:
        """
        Lists all user-defined tables in the main schema of the database.

        Returns:
            List[str]: A list of table names. Returns an empty list if the
                       database file does not exist or on error.
        """
        logger.info(f"Attempting to list tables in {self.db_path}")
        tables = []
        try:
            # Query duckdb_tables information schema view for user tables
            results = self.con.execute(
                "SELECT table_name FROM duckdb_tables WHERE schema_name = 'main' ORDER BY table_name"
            ).fetchall()
            tables = [row[0] for row in results]
            logger.info(f"Found {len(tables)} tables: {tables}")
        except duckdb.IOException as e:
             # Specific case where the file didn't exist and couldn't be created/opened
             logger.error(f"Could not list tables, IO error connecting to {self.db_path}: {e}")
             return [] # Return empty list if DB file is inaccessible
        except duckdb.Error as e:
            logger.error(f"Failed to list tables in {self.db_path}: {e}")
            return [] # Return empty list on other DB errors
        except Exception as e:
            logger.exception(f"An unexpected error occurred while listing tables in {self.db_path}: {e}")
            return [] # Return empty list on unexpected errors

        return tables


    # ──────────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────────
    def _ensure_table(self, table: str, df: pd.DataFrame) -> None:
        """
        Creates the table if absent, or adds any still‑missing columns.
        """
        # 1) Create‑if‑missing with the two key columns
        if "unique_identifier" in df.columns:
            create_cmd = f"""
                CREATE TABLE IF NOT EXISTS "{table}" (
                time_index        TIMESTAMPTZ   NOT NULL,
                unique_identifier VARCHAR   NOT NULL,
                PRIMARY KEY (time_index, unique_identifier)
            )"""
        else:
            create_cmd = f"""
                CREATE TABLE IF NOT EXISTS "{table}" (
                time_index        TIMESTAMPTZ   NOT NULL,
                PRIMARY KEY (time_index)
            )"""

        self.con.execute(create_cmd)

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


